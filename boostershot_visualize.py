import argparse
import cv2
import kornia
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as T

from multiview_detector.models.boostershot import BoosterSHOT
from multiview_detector.utils import projection
from multiview_detector.utils.image_utils import img_color_denormalize, array2heatmap, add_heatmap_to_image
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.datasets import *


def main(args):
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # dataset
    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.expanduser('/workspace/Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser('/workspace/Data/MultiviewX'))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')

    test_set = frameDataset(base, train=False, world_reduce=args.world_reduce,
                            img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                            img_kernel_size=args.img_kernel_size)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4,
                             pin_memory=True, worker_init_fn=seed_worker)

    # model and checkpoint loading
    model = BoosterSHOT(test_set, args.arch, world_feat_arch=args.world_feat,
                    bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, dropout=args.dropout, depth_scales=args.depth_scales).cuda()
    ckpt = torch.load(os.path.join(args.ckpt_path, 'MultiviewDetector.pth'))
    model.load_state_dict(ckpt)

    # create directory
    os.makedirs(os.path.join(args.ckpt_path, 'visualize'), exist_ok=True)
    os.makedirs(os.path.join(args.ckpt_path, 'heatmap'), exist_ok=True)
    os.makedirs(os.path.join(args.ckpt_path, 'separate'), exist_ok=True)

    heatmap = torch.zeros((base.num_cam, args.depth_scales, args.bottleneck_dim))
    
    for batch_idx, (data, _, _, _, _) in enumerate(test_loader):
        print(f'Running batch {batch_idx + 1}')
        B, N, C, H, W = data.shape
        data = data.view(B * N, C, H, W).cuda()

        feat = model.base(data)
        bottleneck = model.bottleneck(feat)
        N, C, _, _ = bottleneck.shape
        block_size = C // args.depth_scales


        if batch_idx == 0:
            cutoff = model.cutoff(bottleneck)
            for i in range(args.depth_scales):
                in_feat = cutoff[:, block_size*i:block_size*(i+1), :, :]
                pooled_feat = torch.mean(in_feat, 1).unsqueeze(1)
                attn_map = model.spatial_attn[f'{i}'].compress(in_feat)
                attn_map = model.spatial_attn[f'{i}'].spatial(attn_map)
                attn_map = F.sigmoid(attn_map)
                pooled_map = attn_map.squeeze(1)
                for cam in range(N):
                    img = denormalize(data[cam]).detach().cpu().numpy().squeeze().transpose([1, 2, 0])
                    img = Image.fromarray((img * 255).astype('uint8'))
                    heatmap_img = torch.norm(pooled_feat[cam].detach(), dim=0).cpu().numpy()
                    visualize_img = add_heatmap_to_image(heatmap_img, img)
                    visualize_img.save(os.path.join(args.ckpt_path, f'visualize/channel_select_{i}_{cam + 1}.png'))
                    heatmap_img = array2heatmap(heatmap_img)
                    heatmap_img.save(os.path.join(args.ckpt_path, f'separate/channel_select_{i}_{cam + 1}.png'))

                    heatmap_img = torch.norm(attn_map[cam].detach(), dim=0).cpu().numpy()
                    visualize_img = add_heatmap_to_image(heatmap_img, img)
                    visualize_img.save(os.path.join(args.ckpt_path, f'visualize/spatial_attn_{i}_{cam + 1}.png'))
                    heatmap_img = array2heatmap(heatmap_img)
                    heatmap_img.save(os.path.join(args.ckpt_path, f'separate/spatial_attn_{i}_{cam + 1}.png'))

        attn = model.cutoff.channel_attn(bottleneck)
        values, indices = torch.topk(attn, block_size, dim=1)
        indices = indices.squeeze(-2).squeeze(-2).detach().cpu()
        
        for cam in range(base.num_cam):
            for i in range(args.depth_scales):
                heatmap[cam, i, :] += torch.zeros(heatmap.shape[-1]).index_fill_(0, indices[cam, :, i], 1)
    
    heatmap = heatmap / float(len(test_loader))
    for cam in range(base.num_cam):
        visualize_img = heatmap[cam].detach().cpu().numpy()
        figure, axes = plt.subplots()
        figure.set_size_inches(48, 8)
        x = np.arange(1, args.bottleneck_dim + 1, 1)
        y = np.arange(0, args.depth_scales, 1)
        color = axes.pcolormesh(x, y, heatmap[cam], vmin=0, vmax=1, cmap='cividis')
        axes.set_yticks(y, fontsize=60)
        axes.set_xticks(x, minor=True)
        axes.set_xticks([32, 64, 96, 128], fontsize=60)
        axes.set_xticklabels([32, 64, 96, 128], fontsize=60)
        axes.set_yticklabels(y ,fontsize=60)
        plt.xlabel('Channels', fontsize=50)
        plt.ylabel('Homographies', fontsize=50)
        plt.axhline(y = 0.5, color = 'w')
        plt.axhline(y = 1.5, color = 'w')
        plt.axhline(y = 2.5, color = 'w')
        plt.grid(axis='x', color='white', which='minor')
        # cbar = figure.colorbar(color)
        # cbar.ax.tick_params(labelsize=30)
        plt.tight_layout()
        plt.savefig(os.path.join(args.ckpt_path, f'heatmap/selection_heatmap_{cam + 1}.png'))

    pass        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--ckpt_path', help='Absolute path of parent directory')
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'mobilenet'])
    parser.add_argument('--world_feat', type=str, default='deform_trans',
                        choices=['conv', 'trans', 'deform_conv', 'deform_trans', 'aio'])
    parser.add_argument('--depth_scales', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument('--augmentation', type=str2bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropcam', type=float, default=0.0)

    parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)
    
    args = parser.parse_args()
    main(args)