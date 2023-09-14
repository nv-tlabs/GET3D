# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import persistence
import nvdiffrast.torch as dr
from training.sample_camera_distribution import sample_camera, create_camera_from_angle
from uni_rep.rep_3d.dmtet import DMTetGeometry
from uni_rep.rep_3d.flexicubes_geometry import FlexiCubesGeometry
from uni_rep.camera.perspective_camera import PerspectiveCamera
from uni_rep.render.neural_render import NeuralRender
from training.discriminator_architecture import Discriminator
from training.geometry_predictor import Conv3DImplicitSynthesisNetwork, TriPlaneTex, \
    MappingNetwork, ToRGBLayer, TriPlaneTexGeo


@persistence.persistent_class
class DMTETSynthesisNetwork(torch.nn.Module):
    def __init__(
            self,
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output image resolution.
            img_channels,  # Number of color channels.
            device='cuda',
            data_camera_mode='carla',
            geometry_type='normal',
            tet_res=64,  # Resolution for tetrahedron grid
            render_type='neural_render',  # neural type
            use_tri_plane=False,
            n_views=2,
            tri_plane_resolution=128,
            deformation_multiplier=2.0,
            feat_channel=128,
            mlp_latent_channel=256,
            dmtet_scale=1.8,
            inference_noise_mode='random',
            one_3d_generator=False,
            iso_surface='dmtet',
            **block_kwargs,  # Arguments for SynthesisBlock.
    ):  #
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.device = device
        self.one_3d_generator = one_3d_generator
        self.inference_noise_mode = inference_noise_mode
        self.dmtet_scale = dmtet_scale
        self.deformation_multiplier = deformation_multiplier
        if iso_surface == "flexicubes":
            self.deformation_multiplier *= 2
        self.geometry_type = geometry_type

        self.data_camera_mode = data_camera_mode
        self.n_freq_posenc_geo = 1
        self.render_type = render_type

        dim_embed_geo = 3 * self.n_freq_posenc_geo * 2
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.n_views = n_views
        self.grid_res = tet_res
        self.iso_surface = iso_surface
        
        # Camera defination, we follow the defination from Blender (check the render_shapenet_data/rener_shapenet.py for more details)
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=self.device)

        # Renderer we used.
        dmtet_renderer = NeuralRender(device, camera_model=dmtet_camera)

        # Geometry class for DMTet
        if self.iso_surface == 'dmtet':
            self.dmtet_geometry = DMTetGeometry(
                grid_res=self.grid_res, scale=self.dmtet_scale, renderer=dmtet_renderer, render_type=render_type,
                device=self.device)
        elif self.iso_surface == 'flexicubes':
            self.dmtet_geometry = FlexiCubesGeometry(
                grid_res=self.grid_res, scale=self.dmtet_scale, renderer=dmtet_renderer, render_type=render_type,
                device=self.device)

        self.feat_channel = feat_channel
        self.mlp_latent_channel = mlp_latent_channel
        self.use_tri_plane = use_tri_plane
        if self.one_3d_generator:
            # Use a unified generator for both geometry and texture generation
            shape_min, shape_max = self.dmtet_geometry.getAABB()
            shape_min = shape_min.min()
            shaape_lenght = shape_max.max() - shape_min
            self.generator = TriPlaneTexGeo(
                w_dim=w_dim,
                img_channels=self.feat_channel,
                shape_min=shape_min,
                shape_lenght=shaape_lenght,
                tri_plane_resolution=tri_plane_resolution,
                device=self.device,
                mlp_latent_channel=self.mlp_latent_channel,
                iso_surface=self.iso_surface,
                **block_kwargs)
        else:
            # Use convolutional 3D for geometry generation
            if self.geometry_type == 'conv3d':
                shape_min, shape_max = self.dmtet_geometry.getAABB()
                shape_min = shape_min.min()
                shaape_lenght = shape_max.max() - shape_min
                self.geometry_synthesis_sdf = Conv3DImplicitSynthesisNetwork(
                    shape_min=shape_min,
                    shape_lenght=shaape_lenght,
                    out_channels=1,
                    n_layers=3,
                    w_dim=self.w_dim,
                    voxel_resolution=32,
                    input_channel=dim_embed_geo,
                    device=self.device)  # use 1x1 cov to improve it (toRGB)
                self.geometry_synthesis_def = Conv3DImplicitSynthesisNetwork(
                    shape_min=shape_min,
                    shape_lenght=shaape_lenght,
                    out_channels=3,
                    n_layers=2,
                    w_dim=self.w_dim,
                    voxel_resolution=32,
                    device=self.device,
                    input_channel=dim_embed_geo)
            else:
                raise NotImplementedError

            # Use triplane for texture field generation
            if self.use_tri_plane:
                shape_min, shape_max = self.dmtet_geometry.getAABB()
                shape_min = shape_min.min()
                shaape_lenght = shape_max.max() - shape_min
                self.geometry_synthesis_tex = TriPlaneTex(
                    w_dim=w_dim, img_channels=self.feat_channel,
                    shape_min=shape_min,
                    shape_lenght=shaape_lenght,
                    tri_plane_resolution=tri_plane_resolution,
                    device=self.device,
                    mlp_latent_channel=self.mlp_latent_channel,
                    **block_kwargs)
            else:
                raise NotImplementedError

        self.channels_last = False
        if self.feat_channel > 3:
            # Final layer to convert the texture field to RGB color, this is only a fully connected layer.
            self.to_rgb = ToRGBLayer(
                self.feat_channel, self.img_channels, w_dim=w_dim,
                conv_clamp=256, channels_last=self.channels_last, device=self.device)
        self.glctx = None

    def transform_points(self, p, for_geo=False):
        pi = np.pi
        assert for_geo
        L = self.n_freq_posenc_geo
        p_transformed = torch.cat(
            [torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def get_sdf_deformation_prediction(
            self, ws, position=None,
            sdf_feature=None):
        '''
        Predict SDF and deformation for tetrahedron vertices
        :param ws: latent code for the geometry
        :param position: the location of tetrahedron vertices
        :param sdf_feature: triplane feature map for the geometry
        :return:
        '''
        weight = None
        if position is None:
            init_position = self.dmtet_geometry.verts.unsqueeze(dim=0)
        else:
            init_position = position
        
        # Step 1: predict the SDF and deformation
        if self.one_3d_generator:
            if self.iso_surface == 'flexicubes':
                sdf, deformation, weight = self.generator.get_sdf_def_prediction(
                sdf_feature, ws_geo=ws,
                position=init_position.expand(ws.shape[0], -1, -1),
                flexicubes_indices=self.dmtet_geometry.indices)
            else:
                sdf, deformation = self.generator.get_sdf_def_prediction(
                sdf_feature, ws_geo=ws,
                position=init_position.expand(ws.shape[0], -1, -1))
        else:
            # Position encoding
            transformed_pos = self.transform_points(init_position, for_geo=True).expand(
                ws.shape[0], -1, -1)
            if self.geometry_type == 'conv3d':
                if position is None:
                    deformation = self.geometry_synthesis_def(
                        ws, transformed_pos.expand(ws.shape[0], -1, -1),
                        init_position.expand(ws.shape[0], -1, -1))
                else:
                    deformation = torch.zeros_like(init_position)  # Here we don't run through network
                sdf = self.geometry_synthesis_sdf(
                    ws, transformed_pos.expand(ws.shape[0], -1, -1),
                    init_position.expand(ws.shape[0], -1, -1))
            else:
                raise NotImplementedError

        # Step 2: Normalize the deformation to avoid the flipped triangles.
        deformation = 1.0 / (self.grid_res * self.deformation_multiplier) * torch.tanh(deformation)
        sdf_reg_loss = torch.zeros(sdf.shape[0], device=sdf.device, dtype=torch.float32)

        ####
        # Step 3: Fix some sdf if we observe empty shape (full positive or full negative)
        if self.iso_surface == 'flexicubes':
            sdf_bxnxnxn = sdf.reshape((sdf.shape[0], self.grid_res + 1, self.grid_res + 1, self.grid_res + 1))
            sdf_less_boundary = sdf_bxnxnxn[:, 1:-1, 1:-1, 1:-1].reshape(sdf.shape[0], -1)
            pos_shape = torch.sum((sdf_less_boundary > 0).int(), dim=-1)
            neg_shape = torch.sum((sdf_less_boundary < 0).int(), dim=-1)
            zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        else:
            pos_shape = torch.sum((sdf.squeeze(dim=-1) > 0).int(), dim=-1)
            neg_shape = torch.sum((sdf.squeeze(dim=-1) < 0).int(), dim=-1)
            zero_surface = torch.bitwise_or(pos_shape == 0, neg_shape == 0)
        if torch.sum(zero_surface).item() > 0:
            update_sdf = torch.zeros_like(sdf[0:1])
            max_sdf = sdf.max()
            min_sdf = sdf.min()
            update_sdf[:, self.dmtet_geometry.center_indices] += (1.0 - min_sdf)  # greater than zero
            update_sdf[:, self.dmtet_geometry.boundary_indices] += (-1 - max_sdf)  # smaller than zero
            new_sdf = torch.zeros_like(sdf)
            for i_batch in range(zero_surface.shape[0]):
                if zero_surface[i_batch]:
                    new_sdf[i_batch:i_batch + 1] += update_sdf
            update_mask = (new_sdf == 0).float()
            # Regulraization here is used to push the sdf to be a different sign (make it not fully positive or fully negative)
            sdf_reg_loss = torch.abs(sdf).mean(dim=-1).mean(dim=-1)
            sdf_reg_loss = sdf_reg_loss * zero_surface.float()
            sdf = sdf * update_mask + new_sdf * (1 - update_mask)

        # Step 4: Here we remove the gradient for the bad sdf (full positive or full negative)
        final_sdf = []
        final_def = []
        for i_batch in range(zero_surface.shape[0]):
            if zero_surface[i_batch]:
                final_sdf.append(sdf[i_batch: i_batch + 1].detach())
                final_def.append(deformation[i_batch: i_batch + 1].detach())
            else:
                final_sdf.append(sdf[i_batch: i_batch + 1])
                final_def.append(deformation[i_batch: i_batch + 1])
        sdf = torch.cat(final_sdf, dim=0)
        deformation = torch.cat(final_def, dim=0)
        return sdf, deformation, sdf_reg_loss, weight

    def get_geometry_prediction(self, ws, sdf_feature=None):
        '''
        Function to generate mesh with give latent code
        :param ws: latent code for geometry generation
        :param sdf_feature: triplane feature for geometry generation
        :return:
        '''

        # Step 1: first get the sdf and deformation value for each vertices in the tetrahedon grid.
        sdf, deformation, sdf_reg_loss, weight = self.get_sdf_deformation_prediction(
            ws,
            sdf_feature=sdf_feature)
        v_deformed = self.dmtet_geometry.verts.unsqueeze(dim=0).expand(sdf.shape[0], -1, -1) + deformation
        tets = self.dmtet_geometry.indices
        n_batch = ws.shape[0]
        v_list = []
        f_list = []
        flexicubes_surface_reg_list = []
        # Step 2: Using marching tet to obtain the mesh
        for i_batch in range(n_batch):
            if self.iso_surface == 'flexicubes':
                verts, faces, flexicubes_surface_reg = self.dmtet_geometry.get_mesh(
                v_deformed[i_batch], sdf[i_batch].squeeze(dim=-1),
                with_uv=False, indices=tets, weight_n=weight[i_batch].squeeze(dim=-1),
                is_training=self.training)     
                flexicubes_surface_reg_list.append(flexicubes_surface_reg)           
            elif self.iso_surface == 'dmtet':
                verts, faces = self.dmtet_geometry.get_mesh(
                v_deformed[i_batch], sdf[i_batch].squeeze(dim=-1),
                with_uv=False, indices=tets)
            v_list.append(verts)
            f_list.append(faces)
        if self.iso_surface == 'flexicubes':
            flexicubes_surface_reg = torch.cat(flexicubes_surface_reg_list).mean()
            flexicubes_weight_reg = (weight ** 2).mean()
        else:
            flexicubes_surface_reg, flexicubes_weight_reg = None, None
        return v_list, f_list, sdf, deformation, v_deformed, (sdf_reg_loss, flexicubes_surface_reg, flexicubes_weight_reg)

    def get_texture_prediction(self, ws, tex_pos, ws_geo=None, hard_mask=None, tex_feature=None):
        '''
        Predict Texture given latent codes
        :param ws: Latent code for texture generation
        :param tex_pos: Position we want to query the texture field
        :param ws_geo: latent code for geometry
        :param hard_mask: 2D silhoueete of the rendered image
        :param tex_feature: the triplane feature map
        :return:
        '''
        tex_pos = torch.cat(tex_pos, dim=0)
        if not hard_mask is None:
            tex_pos = tex_pos * hard_mask.float()
        batch_size = tex_pos.shape[0]
        tex_pos = tex_pos.reshape(batch_size, -1, 3)
        ###################
        # We use mask to get the texture location (to save the memory)
        if hard_mask is not None:
            n_point_list = torch.sum(hard_mask.long().reshape(hard_mask.shape[0], -1), dim=-1)
            sample_tex_pose_list = []
            max_point = n_point_list.max()
            expanded_hard_mask = hard_mask.reshape(batch_size, -1, 1).expand(-1, -1, 3) > 0.5
            for i in range(tex_pos.shape[0]):
                tex_pos_one_shape = tex_pos[i][expanded_hard_mask[i]].reshape(1, -1, 3)
                if tex_pos_one_shape.shape[1] < max_point:
                    tex_pos_one_shape = torch.cat(
                        [tex_pos_one_shape, torch.zeros(
                            1, max_point - tex_pos_one_shape.shape[1], 3,
                            device=tex_pos_one_shape.device, dtype=torch.float32)], dim=1)
                sample_tex_pose_list.append(tex_pos_one_shape)
            tex_pos = torch.cat(sample_tex_pose_list, dim=0)

        if self.one_3d_generator:
            tex_feat = self.generator.get_texture_prediction(tex_feature, tex_pos, ws)
        else:
            if self.use_tri_plane:
                tex_feat = self.geometry_synthesis_tex(
                    ws, ws_geo, tex_pos,
                    noise_mode=self.inference_noise_mode)
            else:
                raise NotImplementedError
        if hard_mask is not None:
            final_tex_feat = torch.zeros(
                ws.shape[0], hard_mask.shape[1] * hard_mask.shape[2], tex_feat.shape[-1], device=tex_feat.device)
            expanded_hard_mask = hard_mask.reshape(hard_mask.shape[0], -1, 1).expand(-1, -1, final_tex_feat.shape[-1]) > 0.5
            for i in range(ws.shape[0]):
                final_tex_feat[i][expanded_hard_mask[i]] = tex_feat[i][:n_point_list[i]].reshape(-1)
            tex_feat = final_tex_feat

        return tex_feat.reshape(ws.shape[0], hard_mask.shape[1], hard_mask.shape[2], tex_feat.shape[-1])

    def generate_random_camera(self, batch_size, n_views=2):
        '''
        Sample a random camera from the camera distribution during training
        :param batch_size: batch size for the generator
        :param n_views: number of views for each shape within a batch
        :return:
        '''
        sample_r = None
        world2cam_matrix, forward_vector, camera_origin, rotation_angle, elevation_angle = sample_camera(
            self.data_camera_mode, batch_size * n_views, self.device)
        mv_batch = world2cam_matrix
        campos = camera_origin
        return campos.reshape(batch_size, n_views, 3), mv_batch.reshape(batch_size, n_views, 4, 4), \
               rotation_angle, elevation_angle, sample_r

    def render_mesh(self, mesh_v, mesh_f, cam_mv):
        '''
        Function to render a generated mesh with nvdiffrast
        :param mesh_v: List of vertices for the mesh
        :param mesh_f: List of faces for the mesh
        :param cam_mv:  4x4 rotation matrix
        :return:
        '''
        return_value_list = []
        for i_mesh in range(len(mesh_v)):
            return_value = self.dmtet_geometry.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=self.img_resolution,
                hierarchical_mask=False
            )
            return_value_list.append(return_value)

        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask_list, hard_mask_list = torch.cat(return_value['mask'], dim=0), torch.cat(return_value['hard_mask'], dim=0)
        return mask_list, hard_mask_list, return_value

    def extract_3d_shape(
            self, ws, ws_geo=None, texture_resolution=2048,
            **block_kwargs):
        '''
        Extract the 3D shape with texture map with GET3D generator
        :param ws: latent code to control texture generation
        :param ws_geo: latent code to control geometry generation
        :param texture_resolution: the resolution for texure map
        :param block_kwargs:
        :return:
        '''

        # Step 1: predict geometry first
        if self.one_3d_generator:
            sdf_feature, tex_feature = self.generator.get_feature(
                ws[:, :self.generator.tri_plane_synthesis.num_ws_tex],
                ws_geo[:, :self.generator.tri_plane_synthesis.num_ws_geo])
            ws = ws[:, self.generator.tri_plane_synthesis.num_ws_tex:]
            ws_geo = ws_geo[:, self.generator.tri_plane_synthesis.num_ws_geo:]
            mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo, sdf_feature)
        else:
            mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo)

        # Step 2: use x-atlas to get uv mapping for the mesh
        from training.extract_texture_map import xatlas_uvmap
        all_uvs = []
        all_mesh_tex_idx = []
        all_gb_pose = []
        all_uv_mask = []
        if self.dmtet_geometry.renderer.ctx is None:
            self.dmtet_geometry.renderer.ctx = dr.RasterizeGLContext(device=self.device)
        for v, f in zip(mesh_v, mesh_f):
            uvs, mesh_tex_idx, gb_pos, mask = xatlas_uvmap(
                self.dmtet_geometry.renderer.ctx, v, f, resolution=texture_resolution)
            all_uvs.append(uvs)
            all_mesh_tex_idx.append(mesh_tex_idx)
            all_gb_pose.append(gb_pos)
            all_uv_mask.append(mask)

        tex_hard_mask = torch.cat(all_uv_mask, dim=0).float()

        # Step 3: Query the texture field to get the RGB color for texture map
        # we use run one per iteration to avoid OOM error
        all_network_output = []
        for _ws, _all_gb_pose, _ws_geo, _tex_hard_mask in zip(ws, all_gb_pose, ws_geo, tex_hard_mask):
            if self.one_3d_generator:
                tex_feat = self.get_texture_prediction(
                    _ws.unsqueeze(dim=0), [_all_gb_pose],
                    _ws_geo.unsqueeze(dim=0).detach(),
                    _tex_hard_mask.unsqueeze(dim=0),
                    tex_feature)
            else:
                tex_feat = self.get_texture_prediction(
                    _ws.unsqueeze(dim=0), [_all_gb_pose],
                    _ws_geo.unsqueeze(dim=0).detach(),
                    _tex_hard_mask.unsqueeze(dim=0))
            background_feature = torch.zeros_like(tex_feat)
            # Merge them together
            img_feat = tex_feat * _tex_hard_mask.unsqueeze(dim=0) + background_feature * (
                    1 - _tex_hard_mask.unsqueeze(dim=0))
            network_out = self.to_rgb(img_feat.permute(0, 3, 1, 2), _ws.unsqueeze(dim=0)[:, -1])
            all_network_output.append(network_out)
        network_out = torch.cat(all_network_output, dim=0)
        return mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out

    def generate_rotate_camera_list(self, n_batch=1):
        '''
        Generate a camera list for rotating the object.
        :param n_batch:
        :return:
        '''
        n_camera = 24
        camera_radius = 1.2  # align with what ww did in blender
        camera_r = torch.zeros(n_camera, 1, device=self.device) + camera_radius
        camera_phi = torch.zeros(n_camera, 1, device=self.device) + (90.0 - 15.0) / 90.0 * 0.5 * math.pi
        camera_theta = torch.range(0, n_camera - 1, device=self.device).unsqueeze(dim=-1) / n_camera * math.pi * 2.0
        camera_theta = -camera_theta
        world2cam_matrix, camera_origin, _, _, _ = create_camera_from_angle(
            camera_phi, camera_theta, camera_r, device=self.device)
        camera_list = [world2cam_matrix[i:i + 1].expand(n_batch, -1, -1).unsqueeze(dim=1) for i in range(n_camera)]
        return camera_list

    def generate(
            self, ws_tex, camera=None, ws_geo=None, **block_kwargs):
        '''
        Main function of our Generator. Given two latent code `ws_tex` for texture generation
        `ws_geo` for geometry generation. It first generate 3D mesh, then render it into 2D image
        with given `camera` or sampled from a prior distribution of camera.
        :param ws_tex: latent code for texture
        :param camera: camera to render generated 3D shape
        :param ws_geo: latent code for geometry
        :param block_kwargs:
        :return:
        '''

        # Generate 3D mesh first
        if self.one_3d_generator:
            sdf_feature, tex_feature = self.generator.get_feature(
                ws_tex[:, :self.generator.tri_plane_synthesis.num_ws_tex],
                ws_geo[:, :self.generator.tri_plane_synthesis.num_ws_geo])
            ws_tex = ws_tex[:, self.generator.tri_plane_synthesis.num_ws_tex:]
            ws_geo = ws_geo[:, self.generator.tri_plane_synthesis.num_ws_geo:]
            mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo, sdf_feature)
        else:
            mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo)

        # Generate random camera
        with torch.no_grad():
            if camera is None:
                campos, cam_mv, rotation_angle, elevation_angle, sample_r = self.generate_random_camera(
                    ws_tex.shape[0], n_views=self.n_views)
                gen_camera = (campos, cam_mv, sample_r, rotation_angle, elevation_angle)
                run_n_view = self.n_views
            else:
                if isinstance(camera, tuple):
                    cam_mv = camera[0]
                    campos = camera[1]
                else:
                    cam_mv = camera
                    campos = None
                gen_camera = camera
                run_n_view = cam_mv.shape[1]

        # Render the mesh into 2D image (get 3d position of each image plane)
        antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv)

        mask_pyramid = None

        tex_pos = return_value['tex_pos']
        tex_hard_mask = hard_mask
        tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
        tex_hard_mask = torch.cat(
            [torch.cat(
                [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
                 for i_view in range(run_n_view)], dim=2)
                for i in range(ws_tex.shape[0])], dim=0)

        # Querying the texture field to predict the texture feature for each pixel on the image
        if self.one_3d_generator:
            tex_feat = self.get_texture_prediction(
                ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask,
                tex_feature)
        else:
            tex_feat = self.get_texture_prediction(
                ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask)
        background_feature = torch.zeros_like(tex_feat)

        # Merge them together
        img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

        # We should split it back to the original image shape
        img_feat = torch.cat(
            [torch.cat(
                [img_feat[i:i + 1, :, self.img_resolution * i_view: self.img_resolution * (i_view + 1)]
                 for i_view in range(run_n_view)], dim=0) for i in range(len(return_value['tex_pos']))], dim=0)

        ws_list = [ws_tex[i].unsqueeze(dim=0).expand(return_value['tex_pos'][i].shape[0], -1, -1) for i in
                   range(len(return_value['tex_pos']))]
        ws = torch.cat(ws_list, dim=0).contiguous()

        # Predict the RGB color for each pixel (self.to_rgb is 1x1 convolution)
        if self.feat_channel > 3:
            network_out = self.to_rgb(img_feat.permute(0, 3, 1, 2), ws[:, -1])
        else:
            network_out = img_feat.permute(0, 3, 1, 2)
        img = network_out
        img_buffers_viz = None

        if self.render_type == 'neural_render':
            img = img[:, :3]
        else:
            raise NotImplementedError

        img = torch.cat([img, antilias_mask.permute(0, 3, 1, 2)], dim=1)
        return img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_buffers_viz, \
               mask_pyramid, tex_hard_mask, sdf_reg_loss, return_value

    def forward(self, ws, camera=None, return_shape=None, **block_kwargs):
        img, antilias_mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, \
        tex_hard_mask, sdf_reg_loss, render_return_value = self.generate(ws, camera, **block_kwargs)
        if return_shape:
            return img, sdf, gen_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, sdf_reg_loss, render_return_value
        return img, gen_camera, mask_pyramid, sdf_reg_loss, render_return_value


@persistence.persistent_class
class GeneratorDMTETMesh(torch.nn.Module):
    def __init__(
            self,
            z_dim,  # Input latent (Z) dimensionality.
            c_dim,  # Conditioning label (C) dimensionality.
            w_dim,  # Intermediate latent (W) dimensionality.
            img_resolution,  # Output resolution.
            img_channels,  # Number of output color channels.
            mapping_kwargs={},  # Arguments for MappingNetwork.
            use_style_mixing=False,  # Whether use stylemixing or not
            **synthesis_kwargs,  # Arguments for SynthesisNetpwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.device = synthesis_kwargs['device']
        self.use_style_mixing = use_style_mixing

        self.synthesis = DMTETSynthesisNetwork(
            w_dim=w_dim, img_resolution=img_resolution, img_channels=self.img_channels,
            **synthesis_kwargs)

        if self.synthesis.one_3d_generator:
            self.num_ws = self.synthesis.generator.num_ws_tex
            self.num_ws_geo = self.synthesis.generator.num_ws_geo
        else:
            self.num_ws = self.synthesis.geometry_synthesis_tex.num_ws_all
            self.num_ws_geo = self.synthesis.geometry_synthesis_sdf.num_ws_all

        self.mapping = MappingNetwork(
            z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws,
            device=self.synthesis.device, **mapping_kwargs)
        self.mapping_geo = MappingNetwork(
            z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws_geo,
            device=self.synthesis.device, **mapping_kwargs)

    def update_w_avg(self, c=None):
        # Update the the average latent to compute truncation
        self.mapping.update_w_avg(self.device, c)
        self.mapping_geo.update_w_avg(self.device, c)

    def generate_3d_mesh(
            self, geo_z, tex_z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False,
            with_texture=True, use_style_mixing=False, use_mapping=True, **synthesis_kwargs):
        '''
        This function generates a 3D mesh with given geometry latent code (geo_z) and texture
        latent code (tex_z), it can also generate a texture map is setting `with_texture` to be True.
        :param geo_z: lantent code for geometry
        :param tex_z: latent code for texture
        :param c: None is default
        :param truncation_psi: the trucation for the latent code
        :param truncation_cutoff: Where to cut the truncation
        :param update_emas: False is default
        :param with_texture: Whether generating texture map along with the 3D mesh
        :param use_style_mixing: Whether use style mixing for generation
        :param use_mapping: Whether we need to use mapping network to map the latent code
        :param synthesis_kwargs:
        :return:
        '''
        if not with_texture:
            self.style_mixing_prob = 0.9
            # Mapping the z to w space
            if use_mapping:
                ws_geo = self.mapping_geo(
                    geo_z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                    update_emas=update_emas)
            else:
                ws_geo = geo_z
            if use_style_mixing:
                cutoff = torch.empty([], dtype=torch.int64, device=ws_geo.device).random_(1, ws_geo.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws_geo.device) < self.style_mixing_prob, cutoff,
                    torch.full_like(cutoff, ws_geo.shape[1]))
                ws_geo[:, cutoff:] = self.mapping_geo(
                    torch.randn_like(geo_z), c, update_emas=False,
                    truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)[:, cutoff:]
            if self.synthesis.one_3d_generator:
                # For this model, we first generate the feature map for it
                ws_tex = self.mapping(geo_z, c, truncation_psi=truncation_psi)  # we didn't use it during inference
                sdf_feature, tex_feature = self.synthesis.generator.get_feature(
                    ws_tex[:, :self.synthesis.generator.tri_plane_synthesis.num_ws_tex],
                    ws_geo[:, :self.synthesis.generator.tri_plane_synthesis.num_ws_geo])
                ws_tex = ws_tex[:, self.synthesis.generator.tri_plane_synthesis.num_ws_tex:]
                ws_geo = ws_geo[:, self.synthesis.generator.tri_plane_synthesis.num_ws_geo:]
                mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.synthesis.get_geometry_prediction(ws_geo, sdf_feature)
            else:
                mesh_v, mesh_f, sdf, deformation, v_deformed, _ = self.synthesis.get_geometry_prediction(ws_geo)
            return mesh_v, mesh_f
        if use_mapping:
            ws = self.mapping(
                tex_z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
            ws_geo = self.mapping_geo(
                geo_z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
                update_emas=update_emas)
        else:
            ws = tex_z
            ws_geo = geo_z

        if use_style_mixing:
            self.style_mixing_prob = 0.9

            cutoff = torch.empty([], dtype=torch.int64, device=ws_geo.device).random_(1, ws_geo.shape[1])
            cutoff = torch.where(
                torch.rand([], device=ws_geo.device) < self.style_mixing_prob, cutoff,
                torch.full_like(cutoff, ws_geo.shape[1]))
            ws_geo[:, cutoff:] = self.mapping_geo(
                torch.randn_like(geo_z), c, update_emas=False,
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff)[:, cutoff:]

            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(
                torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.mapping(
                torch.randn_like(tex_z), c, update_emas=False, truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff
            )[:,
                             cutoff:]
        all_mesh = self.synthesis.extract_3d_shape(ws, ws_geo, )

        return all_mesh

    def generate_3d(
            self, z, geo_z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, camera=None,
            generate_no_light=False,
            **synthesis_kwargs):

        ws = self.mapping(
            z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        ws_geo = self.mapping_geo(
            geo_z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
            update_emas=update_emas)
        img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, mask_pyramid, tex_hard_mask, \
        sdf_reg_loss, render_return_value = self.synthesis.generate(
            ws, camera=camera,
            ws_geo=ws_geo,
            **synthesis_kwargs)
        if generate_no_light:
            return img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask
        return img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, tex_hard_mask

    def forward(
            self, z=None, c=None, truncation_psi=1, truncation_cutoff=None, update_emas=False, use_style_mixing=False,
            geo_z=None,
            **synthesis_kwargs):
        '''
        The function generate rendered 2D image of 3D shapes using the given sampled z for texture and geometry
        :param z:  sample z for textur generation
        :param c: None is default
        :param truncation_psi: truncation value
        :param truncation_cutoff: where to cut the truncation
        :param update_emas: False is default
        :param use_style_mixing: whether use style-mixing
        :param geo_z: sample z for geometry generation
        :param synthesis_kwargs:
        :return:
        '''
        ws = self.mapping(
            z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

        if geo_z is None:
            geo_z = torch.randn_like(z)
        ws_geo = self.mapping_geo(
            geo_z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
            update_emas=update_emas)

        if use_style_mixing:
            self.style_mixing_prob = 0.9
            # We have randomization here to make it have different styles
            cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
            cutoff = torch.where(
                torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                torch.full_like(cutoff, ws.shape[1]))
            ws[:, cutoff:] = self.mapping(
                torch.randn_like(z), c, update_emas=False, truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff
            )[:, cutoff:]

            cutoff = torch.empty([], dtype=torch.int64, device=ws_geo.device).random_(1, ws_geo.shape[1])
            cutoff = torch.where(
                torch.rand([], device=ws_geo.device) < self.style_mixing_prob, cutoff,
                torch.full_like(cutoff, ws_geo.shape[1]))
            ws_geo[:, cutoff:] = self.mapping_geo(
                torch.randn_like(z), c, update_emas=False, truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff
            )[:, cutoff:]

        img, sdf, syn_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, _, _ = self.synthesis(
            ws=ws, update_emas=update_emas,
            return_shape=True,
            ws_geo=ws_geo,
        )
        return img
