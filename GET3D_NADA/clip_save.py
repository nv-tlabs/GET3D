"""
To avoid 'Connection Reset by Peer' error by CLIP library, save checkpoint using torch.jit

Usage
    - $ python scripts/clip_save.py <output_dir> [--test] [--device <device>]
"""


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output_dir', type=str, default='.', help='output checkpoint file directory')
    parser.add_argument('-t', '--test', action='store_true', help='to test checkpoint')
    parser.add_argument('-d', '--device', type=str, default='cuda:0', help='device to use in testing')
    return parser.parse_args()


def main(args):
    import os
    import torch
    import clip

    torch.jit.save(clip.load('ViT-B/32', jit=True)[0], os.path.join(args.output_dir, 'clip-vit-b-32.pt'))
    torch.jit.save(clip.load('ViT-B/16', jit=True)[0], os.path.join(args.output_dir, 'clip-vit-b-16.pt'))
    torch.jit.save(clip.load('RN50', jit=True)[0], os.path.join(args.output_dir, 'clip-cnn.pt'))

    model32, _ = clip.load(os.path.join(args.output_dir, 'clip-vit-b-32.pt'))
    model16, _ = clip.load(os.path.join(args.output_dir, 'clip-vit-b-16.pt'))
    modelcnn, _ = clip.load(os.path.join(args.output_dir, 'clip-cnn.pt'))

    if args.test:
        text = 'rusty car'
        embed32 = model32.encode_text(clip.tokenize(text).to(args.device))
        print(embed32.shape)
        embed16 = model16.encode_text(clip.tokenize(text).to(args.device))
        print(embed16.shape)
        print((embed16 * embed32).sum())


if __name__ == '__main__':
    main(parse_args())
