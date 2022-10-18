# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from uni_rep.render import Renderer

_FG_LUT = None


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(), rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')


def xfm_points(points, matrix, use_python=True):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    out = torch.matmul(torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out


class NeuralRender(Renderer):
    def __init__(self, device='cuda', camera_model=None):
        super(NeuralRender, self).__init__()
        self.device = device
        self.ctx = None
        self.projection_mtx = None
        self.camera = camera_model

    def render_mesh(
            self,
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            camera_mv_bx4x4,
            mesh_v_feat_bxnxd,
            resolution=256,
            spp=1,
            device='cuda',
            hierarchical_mask=False
    ):
        assert not hierarchical_mask
        if self.ctx is None:
            self.ctx = dr.RasterizeCudaContext(device=self.device)

        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        v_pos_clip = self.camera.project(v_pos)  # Projection in the camera

        # Render the image,
        # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
        num_layers = 1
        mask_pyramid = None
        assert mesh_t_pos_idx_fx3.shape[0] > 0  # Make sure we have shapes
        mesh_v_feat_bxnxd = torch.cat([mesh_v_feat_bxnxd, v_pos], dim=-1)  # Concatenate the pos  compute the supervision

        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_t_pos_idx_fx3, [resolution * spp, resolution * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                gb_feat, _ = interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)

        hard_mask = torch.clamp(rast[..., -1:], 0, 1)
        antialias_mask = dr.antialias(
            hard_mask.clone().contiguous(), rast, v_pos_clip,
            mesh_t_pos_idx_fx3)

        depth = gb_feat[..., -2:-1]
        ori_mesh_feature = gb_feat[..., :-4]
        return ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth
