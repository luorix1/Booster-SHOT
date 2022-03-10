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

from multiview_detector.models.attnchannelcutoff import AttnChannelCutoff
from multiview_detector.models.channelcutoff import ChannelCutoff
from multiview_detector.utils import projection
from multiview_detector.utils.decode import ctdet_decode, mvdet_decode
from multiview_detector.utils.image_utils import draw_umich_gaussian, img_color_denormalize, array2heatmap, add_heatmap_to_image
from multiview_detector.utils.nms import nms
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.datasets import *


def main(args):
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # dataset is set to MultiviewX
    base = MultiviewX(os.path.expanduser('/workspace/Data/MultiviewX'))
    
    test_set = frameDataset(base, train=False, world_reduce=args.world_reduce,
                            img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                            img_kernel_size=args.img_kernel_size)

    grid_size = test_set.Rworld_shape

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4,
                             pin_memory=True, worker_init_fn=seed_worker)

    # model and checkpoint loading
    model = AttnChannelCutoff(test_set, args.arch, world_feat_arch=args.world_feat,
                    bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, dropout=args.dropout, depth_scales=args.depth_scales).cuda()
    ckpt = torch.load(os.path.join(args.ckpt_path, 'MultiviewDetector.pth'))
    model.load_state_dict(ckpt)

    # create directory
    os.makedirs(os.path.join(args.ckpt_path, 'diagrams'), exist_ok=True)


    results = np.loadtxt(os.path.join(args.ckpt_path, 'test.txt'))
    
    for batch_idx, (data, world_gt, imgs_gt, affine_mats, frame) in enumerate(test_loader):
        if batch_idx == 4:
            break
        
        heatmap = torch.zeros((base.num_cam, args.depth_scales, args.bottleneck_dim))

        H, W = test_set.Rworld_shape
        heatmap = np.zeros([1, H, W], dtype=np.float32)

        res_map_grid = results[results[:, 0] == frame.detach().cpu().numpy(), 1:]

        for result in res_map_grid:
            ct = np.array([result[0] / test_set.world_reduce, result[1] / test_set.world_reduce], dtype=np.float32)
            if 0 <= ct[0] < W and 0 <= ct[1] < H:
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(heatmap[0], ct_int, test_set.world_kernel_size / test_set.world_reduce)

        print(f'Running batch {batch_idx + 1}')
        B, N, C, H, W = data.shape
        (world_heatmap, world_offset), (imgs_heatmap, imgs_offset, imgs_wh)  = model(data.to('cuda:0'), affine_mats)
        
        fig = plt.figure()
        fig.set_size_inches(36, 10)
        
        imgs = data.squeeze(0)

        subplt1 = fig.add_subplot(241)
        subplt2 = fig.add_subplot(242)
        subplt3 = fig.add_subplot(243)
        subplt5 = fig.add_subplot(245)
        subplt6 = fig.add_subplot(246)
        subplt7 = fig.add_subplot(247)
        subplt1.set_xticks([])
        subplt1.set_yticks([])
        subplt2.set_xticks([])
        subplt2.set_yticks([])
        subplt3.set_xticks([])
        subplt3.set_yticks([])
        subplt5.set_xticks([])
        subplt5.set_yticks([])
        subplt6.set_xticks([])
        subplt6.set_yticks([])
        subplt7.set_xticks([])
        subplt7.set_yticks([])

        subplt1.imshow(denormalize(imgs[0]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt2.imshow(denormalize(imgs[1]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt3.imshow(denormalize(imgs[2]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt5.imshow(denormalize(imgs[3]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt6.imshow(denormalize(imgs[4]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))
        subplt7.imshow(denormalize(imgs[5]).detach().cpu().numpy().squeeze().transpose([1, 2, 0]))

        subplt4 = fig.add_subplot(244)
        subplt8 = fig.add_subplot(248)
        
        subplt4.set_xticks([])
        subplt4.set_yticks([])
        subplt8.set_xticks([])
        subplt8.set_yticks([])

        subplt4.imshow(heatmap[0])
        subplt8.imshow(world_gt['heatmap'].squeeze())

        # plt.subplots_adjust(wspace=0.3, hspace=0.05, top=0.9, bottom=0.1, left=0.1, right=0.9)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.savefig(os.path.join(args.ckpt_path, f'diagrams/diagram_{batch_idx + 1}.png'), bbox_inches='tight')


    pass        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization')
    parser.add_argument('--ckpt_path', help='Absolute path of parent directory')
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