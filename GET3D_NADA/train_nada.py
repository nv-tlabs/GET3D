"""
train script of GeneratorGET3DNADA model

Usage
    - $ python train_nada.py --config_path [config_path] --name [exp_name] --suppress
    - $ cat [config_path] | python train_nada.py --pipe --name [exp_name] --suppress

Reference: https://github.com/rinongal/StyleGAN-nada/blob/main/ZSSGAN/train.py
"""

import sys
import os

import time
import tempfile
import yaml
import numpy as np
import torch
from torchvision.utils import save_image
import logging

import dist_util
from model_engine import find_get3d
from nada import GeneratorGET3DNADA
from functional import unfreeze_generator_layers, generate_nada_mode

if find_get3d():
    from torch_utils import custom_ops

SEED = 0
SELECT = 50


def get_logger(exp_name, outdir, rank=0):
    logger = logging.getLogger(exp_name)
    if rank != 0:
        logger.disabled = True
    else:
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging.DEBUG)
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(f'{outdir}/{exp_name}_{time.strftime("%Y-%m-%d-%H-%M", time.gmtime())}.log')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    return logger


def subprocess_fn(rank, config, args, temp_dir):

    if config['GLOBAL']['gpus'] > 1:
        dist_util.setup_dist(temp_dir, rank, config['GLOBAL']['gpus'])

    if rank != 0:
        custom_ops.verbosity = 'none'

    if rank == 0:
        print("STARTING EXPERIENCE WITH NAME : ", args.name)
        print("LOADING GeneratorGET3DNADA")

    with dist_util.synchronized_ops():
        net = GeneratorGET3DNADA(config)
        unfreeze_generator_layers(net.generator_trainable, [], [])

    if dist_util.get_world_size() > 1:
        ddp_net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[dist_util.dev()],
            output_device=dist_util.dev(),
            broadcast_buffers=True,
            bucket_cap_mb=256,
            find_unused_parameters=True,
        )
    else:
        ddp_net = net

    device, outdir, batch, n_vis, sample_1st, sample_2nd, iter_1st, iter_2nd, lr, \
        output_interval, save_interval, gradient_clip_threshold = net.get_loop_settings()

    g_optim = torch.optim.Adam(
        net.generator_trainable.parameters(),
        lr=lr,
        betas=(0.9, 0.99),
    )

    with dist_util.synchronized_ops():
        if rank == 0:
            sample_dir = os.path.join(outdir, "sample")
            ckpt_dir = os.path.join(outdir, "checkpoint")
            os.makedirs(outdir, exist_ok=True)
            os.makedirs(sample_dir, exist_ok=True)
            os.makedirs(ckpt_dir, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    logger = get_logger(args.name, outdir, rank=rank)
    logger.info(f'EXPERIENCE NAME : {args.name} | CONFIG : {args.config_path}  | SEED : {SEED} | BATCH : {batch}')

    z_dim = 512  # Fixed value
    fixed_z_geo = torch.randn(n_vis, z_dim, device=device)  # for eval
    fixed_z_tex = torch.randn(n_vis, z_dim, device=device)
    grid_rows = int(n_vis ** 0.5)

    eval_camera = net.generator_frozen.synthesis.generate_rotate_camera_list(n_batch=1)[4].repeat(n_vis, 1, 1, 1)
    # ------------------ Training 1st --------------

    # latent z should be 2 -> for geo , tex
    # different n_batch latents per gpu <- equals: seeing n_batch * n_gpu latents
    latent_generator = torch.Generator(device)
    latent_generator.manual_seed(rank)
    sample_z_geo = torch.randn(sample_1st, z_dim, device=device, generator=latent_generator)
    sample_z_tex = torch.randn(sample_1st, z_dim, device=device, generator=latent_generator)

    sample_z_geo_chunks = torch.split(sample_z_geo, batch, dim=0)
    sample_z_tex_chunks = torch.split(sample_z_tex, batch, dim=0)
    logger.info(f'START TRAINING LOOP')

    min_loss_store = []

    for epoch in range(iter_1st):
        for i, (z_geo_chunk, z_tex_chunk) in enumerate(zip(sample_z_geo_chunks, sample_z_tex_chunks)):
            # training
            ddp_net.train()

            # memory-efficient forward : support n_view rendering
            _, loss = ddp_net(z_tex_chunk, z_geo_chunk)

            if epoch == iter_1st - 1:  # to choose 50 latents with low loss value
                loss_val = loss.cpu().detach().numpy().tolist()
                min_loss_store += loss_val

            loss = loss.mean()
            ddp_net.zero_grad()
            loss.backward()

            if gradient_clip_threshold == -1:
                pass
            else:
                torch.nn.utils.clip_grad_norm_(net.generator_trainable.parameters(), gradient_clip_threshold)

            g_optim.step()
            logger.info(f'EPOCH : {epoch} | STEP : {i:0>4} | LOSS : {loss:.5f}')

            # evaluation & save results | save checkpoints
            with dist_util.synchronized_ops():
                if rank == 0:
                    if i % output_interval == 0:
                        ddp_net.eval()
                        with torch.no_grad():
                            sampled_dst, _ = generate_nada_mode(
                                net.generator_trainable,
                                fixed_z_tex, fixed_z_geo,
                                use_mapping=True, mode='layer', camera=eval_camera
                            )
                        
                        rgb = sampled_dst[:, :-1]
                        mask = sampled_dst[:, -1:]
                        bg = torch.ones(rgb.shape, device=device)
                        bg *= 0.0001    # for better background 
                        new_dst = rgb*mask + bg*(1-mask)

                        save_image(
                            new_dst,
                            os.path.join(sample_dir, f"Iter1st_Epoch-{epoch}_Step-{i:0>4}.png"),
                            nrow=grid_rows,
                            normalize=True,
                            range=(-1, 1),
                        )
                        logger.info(f'ITER 1st | EPOCH : {epoch} | STEP : {i:0>4} | >> Save images ...')

                    if i % save_interval == 0 and not args.suppress:
                        torch.save( 
                            {
                                "g_ema": net.generator_trainable.state_dict(),
                                "g_optim": g_optim.state_dict(),
                            },
                            f"{ckpt_dir}/Iter1st_Epoch-{epoch}_Step-{i:0>4}.pt",
                        )
                        logger.info(f'ITER 1st | EPOCH : {epoch} | STEP : {i:0>4} | >> Save checkpoint ...')

            torch.cuda.empty_cache()

            dist_util.barrier()

    logger.info(f"SELCT TOP {SELECT} Latents")
    min_topk_val, min_topk_idx = torch.topk(torch.tensor(min_loss_store), SELECT, largest=False)
    print("SELECT : ", min_topk_val, min_topk_idx)

    # ------------------ Training 2nd --------------

    selected_z_geo = sample_z_geo[min_topk_idx]
    selected_z_tex = sample_z_tex[min_topk_idx]

    selected_z_geo_chunks = torch.split(selected_z_geo, batch, dim=0)
    selected_z_tex_chunks = torch.split(selected_z_tex, batch, dim=0)

    min_loss = 1000

    for epoch in range(iter_2nd):
        for i, (z_geo_chunk, z_tex_chunk) in enumerate(zip(selected_z_geo_chunks, selected_z_tex_chunks)):
            # training
            ddp_net.train()

            _, loss = ddp_net(z_tex_chunk, z_geo_chunk)
            
            loss = loss.mean()
            ddp_net.zero_grad()
            loss.backward()

            if gradient_clip_threshold == -1:
                pass
            else:
                torch.nn.utils.clip_grad_norm_(net.generator_trainable.parameters(), gradient_clip_threshold)

            logger.info(f'ITER 2nd | EPOCH : {epoch} | STEP : {i:0>4} | LOSS : {loss:.5f}')

            # evaluation & save results | save checkpoints
            with dist_util.synchronized_ops():
                if rank == 0:
                    if (i == len(selected_z_geo_chunks) - 1) and (epoch == iter_2nd - 1):
                        torch.save(
                            {
                                "g_ema": net.generator_trainable.state_dict(),
                                "g_optim": g_optim.state_dict(),
                            },
                            f"{ckpt_dir}/latest.pt",
                        )          

                    if i % output_interval == 0:
                        ddp_net.eval()

                        with torch.no_grad():
                            sampled_dst, _ = generate_nada_mode(
                                net.generator_trainable,
                                fixed_z_tex, fixed_z_geo, use_mapping=True, mode='layer', camera=eval_camera
                            )

                        rgb = sampled_dst[:, :-1]
                        mask = sampled_dst[:, -1:]
                        bg = torch.ones(rgb.shape, device=device)
                        bg *= 0.0001    # for better background 
                        new_dst = rgb*mask + bg*(1-mask)

                        save_image(
                            new_dst,
                            os.path.join(sample_dir, f"Iter2nd_Epoch-{epoch}_Step-{i:0>4}.png"),
                            nrow=grid_rows,
                            normalize=True,
                            range=(-1, 1),
                        )

                        logger.info(f'ITER 2nd | EPOCH : {epoch} | STEP : {i:0>4} | >> Save images ...')

                    if i % save_interval == 0:
                        if not args.suppress:
                            torch.save(
                                {
                                    "g_ema": net.generator_trainable.state_dict(),
                                    "g_optim": g_optim.state_dict(),
                                },
                                f"{ckpt_dir}/Iter2nd_Epoch-{epoch}_Step-{i:0>4}.pt",
                            )

                            logger.info(f'ITER 2nd | EPOCH : {epoch} | STEP : {i:0>4} | >> Save checkpoint ...')

                        if loss < min_loss:
                            min_loss = loss
                            torch.save(
                                    {
                                        "g_ema": net.generator_trainable.state_dict(),
                                        "g_optim": g_optim.state_dict(),
                                    },
                                    f"{ckpt_dir}/best.pt",
                                )

            torch.cuda.empty_cache()
            dist_util.barrier()

    logger.info("TRAINING DONE ...")

    # Check final results
    with dist_util.synchronized_ops():
        if rank == 0:
            net.eval()

            with torch.no_grad():
                last_z_geo = torch.randn(n_vis, z_dim, device=device)
                last_z_tex = torch.randn(n_vis, z_dim, device=device)
                sampled_dst, _ = generate_nada_mode(
                    net.generator_trainable,
                    last_z_tex, last_z_geo, use_mapping=True, mode='layer', camera=eval_camera
                )

            save_image(
                sampled_dst,
                os.path.join(sample_dir, "params_latest_images.png"),
                nrow=grid_rows,
                normalize=True,
                range=(-1, 1),
            )

    logger.info("FINISHED")


def launch_training(args):  # Multiprocessing spawning function
    # Load config and parse the number of GPUs.
    if args.pipe:
        config = yaml.safe_load(sys.stdin)
    else:
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)
    gpus = config['GLOBAL']['gpus']

    # In case of single GPU, directly call the training function.
    if gpus == 1:
        subprocess_fn(0, config, args, None)
        return

    # Otherwise, launch processes.
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn', force=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        torch.multiprocessing.spawn(fn=subprocess_fn, args=(config, args, temp_dir), nprocs=gpus)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='experiments/default_dist.yaml')
    parser.add_argument('--name', type=str, default='default_dist')
    parser.add_argument('--pipe', action='store_true', help='read config from stdin instead of file')
    parser.add_argument('--suppress', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    launch_training(parse_args())
