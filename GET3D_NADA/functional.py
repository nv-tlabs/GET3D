"""
Methods (functions) for GET3D generator, required for NADA training and inference
"""
import torch
import nvdiffrast.torch as dr
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from training.networks_get3d import DMTETSynthesisNetwork
    from training.networks_get3d import GeneratorDMTETMesh


# Class GeneratorDMTETMesh
def get_all_generator_layers_dict(self: "GeneratorDMTETMesh"):

    layer_idx_geo = {}
    layer_idx_tex = {}

    tri_plane_blocks = self.synthesis.generator.tri_plane_synthesis.children()

    idx_geo = 0
    idx_tex = 0

    # triplane
    for block in tri_plane_blocks:
        if hasattr(block, 'conv0'):
            layer_idx_geo[idx_geo] = f'b{block.resolution}.conv0'
            idx_geo += 1
        if hasattr(block, 'conv1'):
            layer_idx_geo[idx_geo] = f'b{block.resolution}.conv1'
            idx_geo += 1
        if hasattr(block, 'togeo'):
            layer_idx_geo[idx_geo] = f'b{block.resolution}.togeo'
            idx_geo += 1
        if hasattr(block, 'totex'):
            layer_idx_tex[idx_tex] = f'b{block.resolution}.totex'
            idx_tex += 1

    # mlp_synthesis
    # note that last number = ModuleList index
    layer_idx_tex[idx_tex] = 'mlp_synthesis_tex.0'
    idx_tex += 1
    layer_idx_tex[idx_tex] = 'mlp_synthesis_tex.1'

    layer_idx_geo[idx_geo] = 'mlp_synthesis_geo.0'
    idx_geo += 1
    layer_idx_geo[idx_geo] = 'mlp_synthesis_geo.1'

    return layer_idx_tex, layer_idx_geo


# Class GeneratorDMTETMesh
def freeze_generator_layers(self: "GeneratorDMTETMesh", layer_tex_dict=None, layer_geo_dict=None):
    assert layer_geo_dict is None and layer_tex_dict is None
    self.synthesis.requires_grad_(False)  # all freeze


# Class GeneratorDMTETMesh
def unfreeze_generator_layers(self: "GeneratorDMTETMesh", topk_idx_tex: list, topk_idx_geo: list):
    """
    args
        topk_idx_tex : chosen layers - geo
        topk_idx_geo : chosen layers - tex
        layer_geo_dict , layer_tex_dict : result of get_all_generator_layers()
    """
    if not topk_idx_tex and not topk_idx_geo:
        self.synthesis.generator.tri_plane_synthesis.requires_grad_(True)
        return  # all unfreeze

    layer_tex_dict, layer_geo_dict = get_all_generator_layers_dict(self)

    for idx_tex in topk_idx_tex:
        if idx_tex >= 7:
            # mlp_synthesis_tex
            mlp_name, layer_idx = layer_tex_dict[idx_tex].split('.')
            layer_tex = getattr(self.synthesis.generator.mlp_synthesis_tex, 'layers')[int(layer_idx)]
            layer_tex.requires_grad_(True)
            self.synthesis.generator.mlp_synthesis_tex.layers[int(layer_idx)] = layer_tex

        else:
            # Texture TriPlane
            block_name, layer_name = layer_tex_dict[idx_tex].split('.')
            block = getattr(self.synthesis.generator.tri_plane_synthesis, block_name)
            getattr(block, layer_name).requires_grad_(True)
            setattr(self.synthesis.generator.tri_plane_synthesis, block_name, block)

    for idx_geo in topk_idx_geo:
        if idx_geo >= 20:
            # mlp_synthesis_sdf
            mlp_name, layer_idx = layer_geo_dict[idx_geo].split('.')
            layer_sdf = getattr(self.synthesis.generator.mlp_synthesis_sdf, 'layers')[int(layer_idx)]
            layer_sdf.requires_grad_(True)
            self.synthesis.generator.mlp_synthesis_sdf.layers[int(layer_idx)] = layer_sdf
            # mlp_synthesis_def
            layer_def = getattr(self.synthesis.generator.mlp_synthesis_def, 'layers')[int(layer_idx)]
            layer_def.requires_grad_(True)
            self.synthesis.generator.mlp_synthesis_def.layers[int(layer_idx)] = layer_def

        else:
            # Geometry TriPlane
            block_name, layer_name = layer_geo_dict[idx_geo].split('.')
            block = getattr(self.synthesis.generator.tri_plane_synthesis, block_name)
            getattr(block, layer_name).requires_grad_(True)
            setattr(self.synthesis.generator.tri_plane_synthesis, block_name, block)


# Class :DMTETSynthesisNetwork
def generate_nada_mode_synthesis(
        self: "DMTETSynthesisNetwork",
        ws,
        ws_geo,
        camera=None,
        texture_resolution=2048,
        mode='nada',
        **block_kwargs
):
    """
    mode='thumbnail' : To make thumbnail
    mode='layer'     : To support layer-freezing
    mode='nada'      : To support 1 latent - N views rendering
    """

    # -------------------      generate    ------------------- #

    # (1) Generate 3D mesh first
    # NOTE :
    # this code is shared by 'def generate' and 'def extract_3d_mesh'
    if self.one_3d_generator:
        sdf_feature, tex_feature = self.generator.get_feature(
            ws[:, :self.generator.tri_plane_synthesis.num_ws_tex],
            ws_geo[:, :self.generator.tri_plane_synthesis.num_ws_geo])
        ws = ws[:, self.generator.tri_plane_synthesis.num_ws_tex:]
        ws_geo = ws_geo[:, self.generator.tri_plane_synthesis.num_ws_geo:]
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo, sdf_feature)
    else:
        mesh_v, mesh_f, sdf, deformation, v_deformed, sdf_reg_loss = self.get_geometry_prediction(ws_geo)

    ws_tex = ws

    # (2) Generate random camera
    with torch.no_grad():
        if camera is None:
            # if mode == "nada" or mode == "layer": #js
            if mode == 'nada':
                campos, cam_mv, rotation_angle, elevation_angle, sample_r = self.generate_random_camera(
                    ws_tex.shape[0], n_views=self.n_views)
                gen_camera = (campos, cam_mv, sample_r, rotation_angle, elevation_angle)
                run_n_view = self.n_views
            else:
                campos, cam_mv, rotation_angle, elevation_angle, sample_r = self.generate_random_camera(
                    ws_tex.shape[0], n_views=1)
                gen_camera = (campos, cam_mv, sample_r, rotation_angle, elevation_angle)
                run_n_view = 1
        else:
            if isinstance(camera, tuple):
                cam_mv = camera[0]
                campos = camera[1]
            else:
                cam_mv = camera
                campos = None
            gen_camera = camera
            run_n_view = cam_mv.shape[1]

    # NOTE
    # tex_pos: Position we want to query the texture field || List[(1,1024, 1024,3) * Batch]
    # tex_hard_mask = 2D silhoueete of the rendered image  || Tensor(Batch, 1024, 1024, 1)

    if mode == 'nada':

        antilias_mask = []
        tex_pos = []
        tex_hard_mask = []
        return_value = {'tex_pos': []}

        for idx in range(self.n_views):
            cam = cam_mv[:, idx, :, :].unsqueeze(1)
            antilias_mask_, hard_mask_, return_value_ = self.render_mesh(mesh_v, mesh_f, cam)
            antilias_mask.append(antilias_mask_)
            tex_hard_mask.append(hard_mask_)

            for pos in return_value_['tex_pos']:
                return_value['tex_pos'].append(pos)

        antilias_mask = torch.cat(antilias_mask, dim=0)  # (B*n_view, 1024, 1024, 1)
        tex_hard_mask = torch.cat(tex_hard_mask, dim=0)  # (B*n_view, 1024, 1024, 3)
        tex_pos = return_value['tex_pos']

        ws_tex = ws_tex.repeat(self.n_views, 1, 1)
        ws_geo = ws_geo.repeat(self.n_views, 1, 1)
        tex_feature = tex_feature.repeat(self.n_views, 1, 1, 1)

    else:
        # (3) Render the mesh into 2D image (get 3d position of each image plane)
        antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv)

        tex_pos = return_value['tex_pos']
        tex_hard_mask = hard_mask

        tex_pos = [torch.cat([pos[i_view:i_view + 1] for i_view in range(run_n_view)], dim=2) for pos in tex_pos]
        tex_hard_mask = torch.cat(
            [torch.cat(
                [tex_hard_mask[i * run_n_view + i_view: i * run_n_view + i_view + 1]
                 for i_view in range(run_n_view)], dim=2)
                for i in range(ws_tex.shape[0])], dim=0)

    # (4) Querying the texture field to predict the texture feature for each pixel on the image
    if self.one_3d_generator:
        tex_feat = self.get_texture_prediction(ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask, tex_feature)
    else:
        tex_feat = self.get_texture_prediction(
            ws_tex, tex_pos, ws_geo.detach(), tex_hard_mask)
    background_feature = torch.zeros_like(tex_feat)

    # (5) Merge them together
    img_feat = tex_feat * tex_hard_mask + background_feature * (1 - tex_hard_mask)

    # NOTE : debug -> no need to execute (6)
    # (6) We should split it back to the original image shape

    ws_list = [ws_tex[i].unsqueeze(dim=0).expand(return_value['tex_pos'][i].shape[0], -1, -1) for i in
               range(len(return_value['tex_pos']))]
    ws = torch.cat(ws_list, dim=0).contiguous()

    # (7) Predict the RGB color for each pixel (self.to_rgb is 1x1 convolution)
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

    return_generate = [img, antilias_mask]

    if mode == 'layer' or mode == 'nada':
        return return_generate[0], None

    elif mode == 'thumbnail':
        # ------------------- extract_3d_shape ------------------- #

        del tex_hard_mask
        del tex_feat

        # (8) Use x-atlas to get uv mapping for the mesh
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

        # (9) Query the texture field to get the RGB color for texture map
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

        return_extract_3d_mesh = [mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, network_out]

        return return_generate, return_extract_3d_mesh

    else:
        raise NotImplementedError


# Class: GeneratorDMTETMesh
def generate_nada_mode(
        self: "GeneratorDMTETMesh",
        geo_z, tex_z, c=0, truncation_psi=1, truncation_cutoff=None,
        update_emas=False, use_mapping=False,  # -> generate_3d_mesh
        camera=None,  # -> generate_3d
        mode='thumbnail',
        **synthesis_kwargs):
    """
    Description
    mode='thumbnail' : To make thumbnail
    mode='layer'     : To support layer-freezing
    mode='nada'      : To support 1 latent - N views rendering

    Note :
    this function don't take below as input args
        1. use_style_mixing
        2. generate_no_light
        3. with_texture
        , since they are redundant.

    Return :
        return_generate_3d = [rendered RGB Image, rendered 2D Silhouette image]
        return_generate_3d_mesh = [mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, texture map]
    """

    if use_mapping or mode == 'thumbnail':
        ws = self.mapping(
            tex_z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
            update_emas=update_emas)
        ws_geo = self.mapping_geo(
            geo_z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff,
            update_emas=update_emas)
    else:
        ws = tex_z
        ws_geo = geo_z

    return generate_nada_mode_synthesis(
        self.synthesis,
        ws, ws_geo, camera=camera, mode=mode, **synthesis_kwargs
    )  # custom inference code.
