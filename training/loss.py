# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix


# ----------------------------------------------------------------------------
class Loss:
    def accumulate_gradients(
            self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------
# Regulrarization loss for dmtet
def sdf_reg_loss_batch(sdf, all_edges):
    sdf_f1x6x2 = sdf[:, all_edges.reshape(-1)].reshape(sdf.shape[0], -1, 2)
    mask = torch.sign(sdf_f1x6x2[..., 0]) != torch.sign(sdf_f1x6x2[..., 1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(
        sdf_f1x6x2[..., 0], (sdf_f1x6x2[..., 1] > 0).float()) + \
               torch.nn.functional.binary_cross_entropy_with_logits(
                   sdf_f1x6x2[..., 1], (sdf_f1x6x2[..., 0] > 0).float())
    return sdf_diff


class StyleGAN2Loss(Loss):
    def __init__(
            self, device, G, D, r1_gamma=10, style_mixing_prob=0, pl_weight=0,
            gamma_mask=10, lambda_flexicubes_surface_reg=1, lambda_flexicubes_weights_reg=0.1):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.r1_gamma = r1_gamma
        self.style_mixing_prob = style_mixing_prob
        self.pl_weight = pl_weight
        self.gamma_mask = gamma_mask
        self.lambda_flexicubes_surface_reg = lambda_flexicubes_surface_reg
        self.lambda_flexicubes_weights_reg = lambda_flexicubes_weights_reg

    def run_G(
            self, z, c, update_emas=False, return_shape=False,
    ):
        # Step 1: Map the sampled z code to w-space
        ws = self.G.mapping(z, c, update_emas=update_emas)
        geo_z = torch.randn_like(z)
        ws_geo = self.G.mapping_geo(
            geo_z, c,
            update_emas=update_emas)

        # Step 2: Apply style mixing to the latent code
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                    torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]

                cutoff = torch.empty([], dtype=torch.int64, device=ws_geo.device).random_(1, ws_geo.shape[1])
                cutoff = torch.where(
                    torch.rand([], device=ws_geo.device) < self.style_mixing_prob, cutoff,
                    torch.full_like(cutoff, ws_geo.shape[1]))
                ws_geo[:, cutoff:] = self.G.mapping_geo(torch.randn_like(z), c, update_emas=False)[:, cutoff:]

        # Step 3: Generate rendered image of 3D generated shapes.
        if return_shape:
            img, sdf, syn_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, sdf_reg_loss, render_return_value = self.G.synthesis(
                ws,
                return_shape=return_shape,
                ws_geo=ws_geo,
            )
            return img, sdf, ws, syn_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, ws_geo, sdf_reg_loss, render_return_value
        else:
            img, syn_camera, mask_pyramid, sdf_reg_loss, render_return_value = self.G.synthesis(
                ws, return_shape=return_shape,
                ws_geo=ws_geo)
        return img, ws, syn_camera, mask_pyramid, render_return_value

    def run_D(self, img, c, update_emas=False, mask_pyramid=None):
        logits = self.D(img, c, update_emas=update_emas, mask_pyramid=mask_pyramid)
        return logits

    def accumulate_gradients(
            self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # First generate the rendered image of generated 3D shapes
                gen_img, gen_sdf, _gen_ws, gen_camera, deformation, v_deformed, mesh_v, mesh_f, mask_pyramid, _gen_ws_geo, \
                sdf_reg_loss, render_return_value = self.run_G(
                    gen_z, gen_c, return_shape=True
                )

                camera_condition = None
                if self.G.synthesis.data_camera_mode == 'shapenet_car' or self.G.synthesis.data_camera_mode == 'shapenet_chair' \
                        or self.G.synthesis.data_camera_mode == 'shapenet_motorbike' or self.G.synthesis.data_camera_mode == 'renderpeople' or \
                        self.G.synthesis.data_camera_mode == 'shapenet_plant' or self.G.synthesis.data_camera_mode == 'shapenet_vase' or \
                        self.G.synthesis.data_camera_mode == 'ts_house' or self.G.synthesis.data_camera_mode == 'ts_animal' or \
                        self.G.synthesis.data_camera_mode == 'all_shapenet':
                    camera_condition = torch.cat((gen_camera[-2], gen_camera[-1]), dim=-1)
                else:
                    assert NotImplementedError
                # Send to discriminator
                gen_logits = self.run_D(gen_img, camera_condition, mask_pyramid=mask_pyramid)
                gen_logits, gen_logits_mask = gen_logits

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits).mean()
                training_stats.report('Loss/G/loss_rgb', loss_Gmain)

                training_stats.report('Loss/scores/fake_mask', gen_logits_mask)
                training_stats.report('Loss/signs/fake_mask', gen_logits_mask.sign())
                loss_Gmask = torch.nn.functional.softplus(-gen_logits_mask).mean()
                training_stats.report('Loss/G/loss_mask', loss_Gmask)
                loss_Gmain += loss_Gmask
                training_stats.report('Loss/G/loss', loss_Gmain)

                # Regularization loss for sdf prediction
                sdf_reg_loss_entropy = sdf_reg_loss_batch(gen_sdf, self.G.synthesis.dmtet_geometry.all_edges).mean() * 0.01
                training_stats.report('Loss/G/sdf_reg', sdf_reg_loss_entropy)
                loss_Gmain += sdf_reg_loss_entropy
                if self.G.synthesis.iso_surface == 'flexicubes':
                    edge_sdf_reg_loss, flexicubes_surface_reg, flexicubes_weights_reg = sdf_reg_loss
                    training_stats.report('Loss/G/sdf_reg_abs', edge_sdf_reg_loss)
                    training_stats.report('Loss/G/flexicubes_surface_reg', flexicubes_surface_reg.mean())
                    training_stats.report('Loss/G/flexicubes_weights_reg', flexicubes_weights_reg.mean())
                    loss_Gmain += edge_sdf_reg_loss.mean() 
                    loss_Gmain += flexicubes_surface_reg.mean() * self.lambda_flexicubes_surface_reg
                    loss_Gmain += flexicubes_weights_reg.mean() * self.lambda_flexicubes_weights_reg
                else:
                    edge_sdf_reg_loss, _, _ = sdf_reg_loss
                    training_stats.report('Loss/G/sdf_reg_abs', edge_sdf_reg_loss)
                    loss_Gmain += edge_sdf_reg_loss.mean()

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # We didn't have Gpl regularization

        #######################################################
        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                # First generate the rendered image of generated 3D shapes
                gen_img, _gen_ws, gen_camera, mask_pyramid, render_return_value = self.run_G(
                    gen_z, gen_c, update_emas=True)
                if self.G.synthesis.data_camera_mode == 'shapenet_car' or self.G.synthesis.data_camera_mode == 'shapenet_chair' \
                        or self.G.synthesis.data_camera_mode == 'shapenet_motorbike' or self.G.synthesis.data_camera_mode == 'renderpeople' or \
                        self.G.synthesis.data_camera_mode == 'shapenet_plant' or self.G.synthesis.data_camera_mode == 'shapenet_vase' or \
                        self.G.synthesis.data_camera_mode == 'ts_house' or self.G.synthesis.data_camera_mode == 'ts_animal' or \
                        self.G.synthesis.data_camera_mode == 'all_shapenet':
                    camera_condition = torch.cat((gen_camera[-2], gen_camera[-1]), dim=-1)
                else:
                    camera_condition = None

                # Send it to discriminator
                gen_logits = self.run_D(
                    gen_img, camera_condition, update_emas=True, mask_pyramid=mask_pyramid)

                gen_logits, gen_logits_mask = gen_logits

                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits).mean()  # -log(1 - sigmoid(gen_logits))
                training_stats.report('Loss/D/loss_genrgb', loss_Dgen)

                training_stats.report('Loss/scores/fake_mask', gen_logits_mask)
                training_stats.report('Loss/signs/fake_mask', gen_logits_mask.sign())
                loss_Dgen_mask = torch.nn.functional.softplus(
                    gen_logits_mask).mean()  # -log(1 - sigmoid(gen_logits))
                training_stats.report('Loss/D/loss_gen_mask', loss_Dgen_mask)
                loss_Dgen += loss_Dgen_mask

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                # Optimize for the real image
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])

                real_logits = self.run_D(real_img_tmp, real_c, )
                real_logits, real_logits_mask = real_logits

                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                training_stats.report('Loss/scores/real_mask', real_logits_mask)
                training_stats.report('Loss/signs/real_mask', real_logits_mask.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits).mean()  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_real_rgb', loss_Dreal)

                    loss_Dreal_mask = torch.nn.functional.softplus(
                        -real_logits_mask).mean()  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss_real_mask', loss_Dreal_mask)
                    loss_Dreal += loss_Dreal_mask
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                # Compute R1 regularization for discriminator
                if phase in ['Dreg', 'Dboth']:
                    # Compute R1 regularization for discriminator of RGB image
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(
                            outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]

                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty.mean() * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)
                    # Compute R1 regularization for discriminator of Mask image
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads_mask = \
                            torch.autograd.grad(
                                outputs=[real_logits_mask.sum()], inputs=[real_img_tmp], create_graph=True,
                                only_inputs=True)[0]

                    r1_penalty_mask = r1_grads_mask.square().sum([1, 2, 3])
                    loss_Dr1_mask = r1_penalty_mask.mean() * (self.gamma_mask / 2)
                    training_stats.report('Loss/r1_penalty_mask', r1_penalty_mask)
                    training_stats.report('Loss/D/reg_mask', loss_Dr1_mask)
                    loss_Dr1 += loss_Dr1_mask
            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()
