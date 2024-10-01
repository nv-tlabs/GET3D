# Copyright (c) 2023 kdha0727
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Helpers for distributed training.
"""

import os
import functools
import contextlib

import torch
import torch.distributed as dist
from torch.cuda import is_available as _cuda_available


RANK = 0
WORLD_SIZE = 1


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                       Setup Tools                                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def is_initialized():
    # if pytorch isn't compiled with c10d, is_initialized is omitted from namespace.
    # this function wraps
    """
    Returns c10d (distributed) runtime is initialized.
    """
    return dist.is_available() and getattr(dist, "is_initialized", lambda: False)()


def setup_dist(temp_dir, rank, world_size):
    """
    Set up a distributed process group.
    """

    if is_initialized():
        return True

    init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
    if os.name == 'nt':
        init_method = 'file:///' + init_file.replace('\\', '/')
        dist.init_process_group(
            backend='gloo', init_method=init_method, rank=rank, world_size=world_size)
    else:
        init_method = f'file://{init_file}'
        dist.init_process_group(
            backend='nccl', init_method=init_method, rank=rank, world_size=world_size)

    global RANK, WORLD_SIZE
    RANK = rank
    WORLD_SIZE = world_size

    torch.cuda.set_device(dev())
    torch.cuda.empty_cache()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                                      General Tools                                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

@functools.lru_cache(maxsize=None)
def get_rank(group=None):
    if group is not None and is_initialized():
        return dist.get_rank(group=group)
    return RANK


@functools.lru_cache(maxsize=None)
def get_world_size(group=None):
    if group is not None and is_initialized():
        return dist.get_world_size(group=group)
    return WORLD_SIZE


def barrier(*args, **kwargs):
    if is_initialized():
        return dist.barrier(*args, **kwargs)


@contextlib.contextmanager
def synchronized_ops():
    barrier()
    yield
    barrier()
    return


@functools.lru_cache(maxsize=None)
def dev(group=None):
    """
    Get the device to use for torch.distributed.
    """
    if _cuda_available():
        return torch.device(get_rank(group))
    return torch.device("cpu")


def load_state_dict(local_or_remote_path, **kwargs):
    """
    Load a PyTorch file.
    """
    with open(local_or_remote_path, "rb") as f:
        return torch.load(f, **kwargs)


def broadcast(tensor, src=0, group=None, async_op=False):
    """
    Synchronize a Tensor across ranks from {src} rank. (default=0)
    :param tensor: torch.Tensor.
    :param src: source rank to sync params from. default is 0.
    :param group:
    :param async_op:
    """
    if not is_initialized():
        return
    with torch.no_grad():
        dist.broadcast(tensor, src, group=group, async_op=async_op)


def sync_params(params, src=0, group=None, async_op=False):
    """
    Synchronize a sequence of Tensors across ranks from {src} rank. (default=0)
    :param params: Sequence of torch.Tensor.
    :param src: source rank to sync params from. default is 0.
    :param group:
    :param async_op:
    """
    if not is_initialized():
        return
    for p in params:
        broadcast(p, src, group=group, async_op=async_op)
