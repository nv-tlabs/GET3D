"""
Config parser class for efficient PM
"""
import os
import sys
import copy
import yaml
import torch
from contextlib import contextmanager

import dist_util
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from training.networks_get3d import GeneratorDMTETMesh

GET3D_ROOT = None


class Engine(object):
    """Config parser class for efficient management"""
    rank: int
    config: dict
    device: torch.device
    global_kwargs: dict
    G_kwargs: dict
    clip_kwargs: dict

    @classmethod
    def parse_engine_like(cls, engine_like):
        if isinstance(engine_like, cls):  # Engine
            return engine_like
        elif isinstance(engine_like, dict):  # config dict
            return cls(engine_like)
        elif isinstance(engine_like, str) or hasattr(engine_like, '__fspath__'):  # path
            with open(engine_like, 'r') as fp:
                return cls(yaml.safe_load(fp))
        elif hasattr(engine_like, 'read'):  # file-like
            return cls(yaml.safe_load(engine_like))
        raise TypeError

    def __init__(self, config: dict, rank: "int|None" = None):
        self.rank = rank
        self.config = config
        self.parse()

    def parse(self):
        if self.rank is None:
            self.rank = dist_util.get_rank()
        self.device = torch.device('cuda', self.rank)

        # setting : global configuration
        self.global_kwargs = dnnlib.EasyDict(self.config['GLOBAL'])

        # ref) get3d : train_3d.py ln251 - ln320
        # setting : GET3D configuration
        opts = dnnlib.EasyDict(self.config['GET3D'])
        # global
        G_kwargs = self.G_kwargs = dnnlib.EasyDict()
        G_kwargs.device = self.device
        G_kwargs.class_name = 'training.networks_get3d.GeneratorDMTETMesh'
        G_kwargs.img_resolution = opts.img_res  # // reformed
        G_kwargs.img_channels = opts.img_channels  # // reformed
        # mapping network
        G_kwargs.z_dim = opts.latent_dim
        G_kwargs.w_dim = opts.latent_dim
        G_kwargs.c_dim = opts.c_dim   # 0(=None) # NOTE : This can be used for class conditioning ... // reformed
        G_kwargs.mapping_kwargs = dnnlib.EasyDict()
        G_kwargs.mapping_kwargs.num_layers = 8
        # stylegan2 + tri-plane
        G_kwargs.use_style_mixing = opts.use_style_mixing
        G_kwargs.one_3d_generator = opts.one_3d_generator
        G_kwargs.dmtet_scale = opts.dmtet_scale
        G_kwargs.n_implicit_layer = opts.n_implicit_layer
        G_kwargs.feat_channel = opts.feat_channel
        G_kwargs.mlp_latent_channel = opts.mlp_latent_channel
        G_kwargs.deformation_multiplier = opts.deformation_multiplier
        G_kwargs.tri_plane_resolution = opts.tri_plane_resolution
        G_kwargs.n_views = opts.n_views
        G_kwargs.use_tri_plane = opts.use_tri_plane
        G_kwargs.tet_res = opts.tet_res
        # G_kwargs.tet_path = '../data/tets'
        # neural renderer
        G_kwargs.render_type = opts.render_type
        G_kwargs.data_camera_mode = opts.data_camera_mode
        # misc
        G_kwargs.fused_modconv_default = 'inference_only'

        # setting : NADA configuration
        clip_kwargs = self.clip_kwargs = dnnlib.EasyDict(self.config['NADA'])
        clip_kwargs.device = self.device

    def build_get3d_pair(self):
        with at_working_directory(GET3D_ROOT):
            G_ema: "GeneratorDMTETMesh" = dnnlib.util.construct_class_by_name(**self.G_kwargs)
            G_ema.to(self.device).train().requires_grad_(False)

        assert self.global_kwargs['resume_pretrain'] != '', "ASSERTION : Specify pretrained GET3D model"
        if self.rank == 0:
            model_state_dict = torch.load(
                self.global_kwargs['resume_pretrain'],
                map_location=self.device
            )
            G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        dist_util.sync_params(G_ema.parameters(), src=0)
        dist_util.sync_params(G_ema.buffers(), src=0)

        G_ema_frozen: "GeneratorDMTETMesh" = copy.deepcopy(G_ema).eval()
        return G_ema, G_ema_frozen


@contextmanager
def at_working_directory(work_dir):
    """Context manager for changing working directory."""
    prev = os.getcwd()
    try:
        os.chdir(work_dir)
        yield
    finally:
        os.chdir(prev)


def find_get3d():
    """
    This function makes dynamic import of GET3D modules available.
    Officially supported ways:
        1. Locate this module's directory in GET3D directory. (recommended)
        2. Locate GET3D via submodule, by `git submodule sync && git submodule update --init --recursive`.
        3. Set GET3D directory via environment variable `GET3D_ROOT`.
        4. Manually specify GET3D directory in this file, by variable `GET3D_ROOT` (line 21).
    """
    global GET3D_ROOT
    # 1. check if GET3D_ROOT is already set and in sys.path
    if GET3D_ROOT is not None and GET3D_ROOT in sys.path:
        return True
    # 2. check if GET3D modules are already imported and __file__ attribute is available
    try:
        import training.networks_get3d
    except ImportError:
        pass
    if hasattr(sys.modules.get('training.networks_get3d', None), '__file__'):
        GET3D_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(sys.modules['training.networks_get3d'].__file__)))
        return True
    # 3. check if GET3D_ROOT is specified via environment variable, or try to guess
    import importlib
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        GET3D_ROOT,
        os.getenv('GET3D_ROOT', None),
        os.path.dirname(base),
        os.path.join(base, 'GET3D'),
    ]
    for candidate in candidates:  # Try each candidate path in order.
        if candidate is not None and os.path.isdir(os.path.join(candidate, 'training')):
            try:
                sys.path.insert(0, candidate)
                importlib.import_module('training.networks_get3d')
                GET3D_ROOT = candidate
                break
            except ImportError:
                sys.path.pop(0)
    if GET3D_ROOT is None:  # Fail if all candidates failed.
        raise ImportError(
            'Failed to find GET3D root directory. '
            'Please specify the location of GET3D via GET3D_ROOT environment variable.'
        )
    else:
        return True


if find_get3d():
    import dnnlib
