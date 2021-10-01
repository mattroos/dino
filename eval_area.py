# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import json
from pathlib import Path

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits

import torchvision.transforms.functional as F
from torchvision.transforms.functional import InterpolationMode

from dataset_inserted_cow import ObjectsAndBackgroundsDataset, IndependentColorJitter, InsertObjectInBackground

import numpy as np
import matplotlib.pyplot as plt
plt.ion()


## Example shell command to start this training script:
# python eval_linear_scale.py --data_path /Data/DairyTech/Flickr_cows_train_val_sets/ --num_workers 8 2>/dev/null
#   The 2>/dev/null tail is to get rid of warning messages from caffe. See here:
#   https://github.com/pytorch/pytorch/issues/57273

MIN_SCALE = 0.666  # Mimimum image scaling
MAX_SCALE = 1.0    # Maximum image scaling
MEAN = (0.485, 0.456, 0.406)  # ImageNet channel means
STD = (0.229, 0.224, 0.225)   # ImageNet channel standard deviations


def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    # train_transform = pth_transforms.Compose([
    #     # pth_transforms.RandomResizedCrop(224),
    #     pth_transforms.RandomHorizontalFlip(),
    #     pth_transforms.ToTensor(),
    #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])
    # val_transform = pth_transforms.Compose([
    #     pth_transforms.Resize(256, interpolation=3),
    #     pth_transforms.CenterCrop(224),
    #     pth_transforms.ToTensor(),
    #     pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # ])
    
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    # dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
    # # dataset_train = datasets.ImageFolder(args.data_path, transform=train_transform)
    # # dataset_val = datasets.ImageFolder(args.data_path, transform=val_transform)
    
    PATH_BACKGROUNDS = '/Data/DairyTech/Flickr_fields_train_val_sets/'
    PATH_COWS = '/home/mroos/Data/DairyTech/labelme/scaled_segmented_train_val_sets/'
    COW_IMAGE_FILE_ENDSWITH = 'cow_side_seg.tiff'
    JITTER = 0.2  # a number between 0 (no JITTER) and 1 (maximum JITTER)
    SHRINKAGE = 0.5

    transform_train = pth_transforms.Compose([IndependentColorJitter(brightness=JITTER, contrast=JITTER, saturation=JITTER, hue=JITTER/2),
                                              InsertObjectInBackground(224, SHRINKAGE, random_flip=True, channels_first=True, normalize=True)])
    dataset_train = ObjectsAndBackgroundsDataset(os.path.join(PATH_COWS, "train/dummy"),
                                                 os.path.join(PATH_BACKGROUNDS, "train/dummy"),
                                                 pattern_objects=COW_IMAGE_FILE_ENDSWITH,
                                                 pattern_backgrounds=None,
                                                 transform=transform_train)
    
    transform_val = InsertObjectInBackground(224, SHRINKAGE, random_flip=True, channels_first=True, normalize=True)
    dataset_val = ObjectsAndBackgroundsDataset(os.path.join(PATH_COWS, "val/dummy"),
                                               os.path.join(PATH_BACKGROUNDS, "val/dummy"),
                                               pattern_objects=COW_IMAGE_FILE_ENDSWITH,
                                               pattern_backgrounds=None,
                                               transform=transform_val)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building ViT network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknown architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    # ============ building head network ... ============
    if args.head_type=='linear':
        head_estimator = LinearEstimator(embed_dim, args)
    elif args.head_type=='mlp':
        head_estimator = MlpEstimator(embed_dim, args)
    else:
        raise Exception(f'Unknown head_type argument: {args.head_type}')
    head_estimator = head_estimator.cuda()
    head_estimator = nn.parallel.DistributedDataParallel(head_estimator, device_ids=[args.gpu])


    # If only visualizing for existing model, do it...
    if args.view:
        to_restore = {"epoch": 0, "best_loss": 1000., "epoch_best_loss": -1}
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth.tar"),
            state_dict=head_estimator,
        )
        # Visualize
        assert args.view_dataset in ['train', 'val']
        if args.view_dataset=='train':
            visualize_predictions(train_loader, model, head_estimator, args.n_last_blocks, args.avgpool_patchtokens)
        else:
            visualize_predictions(val_loader, model, head_estimator, args.n_last_blocks, args.avgpool_patchtokens)
        sys.exit()


    # Check for output directory and try to create it if it doesn't exist
    if not os.path.exists(args.output_dir):
        try:
            print(f'Output directory {args.output_dir} does not exist.')
        except:
            print(f'Output directory {args.output_dir} does not exist.')
            # using try, and doing nothing if it throws an exception, because seem to bomb multi-GPU training otherwise


    # set optimizer
    optimizer = torch.optim.SGD(
        head_estimator.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_loss": 1000., "epoch_best_loss": -1}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=head_estimator,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_loss = to_restore["best_loss"]
    epoch_best_loss = to_restore["epoch_best_loss"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        print('')
        train_stats = train(model, head_estimator, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            print('')
            test_stats = validate_network(val_loader, model, head_estimator, args.n_last_blocks, args.avgpool_patchtokens)
            print(f"Loss at epoch {epoch} of the network on the {len(dataset_val)} test images: {test_stats['loss']:.3e}")
            if test_stats["loss"] < best_loss:
                epoch_best_loss = epoch
            best_loss = min(best_loss, test_stats["loss"])
            print(f'Lowest loss so far: {best_loss:.3e} at epoch {epoch_best_loss}')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": head_estimator.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "epoch_best_loss": epoch_best_loss,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))

    print("Training of the supervised estimator on frozen features completed.\n"
                "Lowest test loss: {loss:.3e}".format(loss=best_loss))


def scale_batch(imgs, scale_range, pad_constant=0.0, interp_mode=InterpolationMode.BILINEAR):
    '''
    Randomly scales (resizes) a batch of images. Each image in the batch undergoes
    identical resizing. Input images must be square. The resizing is equal in the
    horizontal and vertical dimensions.

    Enlargement (zoom in), scale > 1.0:
        A crop of proportion 1/scale is taken from a random location in the image
        and then resized to the original image size.

    Shrinkage (zoom out), scale < 1.0:
        The image is resized to scale factor of the original, and then inserted into
        a random location in an "empty" image equal in size to that of the original.
        The empty/padded values are a constant value.

    Inputs:
        imgs - Tensor of size (batch, channel, height, width).
               The height must equal the width.
        scale_range - The (min, max) scaling values, where 1.0 indicates
               no scaling/resizing. Numbers > 1.0 indicate "zooming in".
        pad_constant - Constant fill value for all three color channels. This doesn't
               currently work as wanted, because RGB normalization has already
               been done, so "black" has a different value for each channel.
    Outputs:
        imgs_scaled - Tensor of resized images, equal in size to that of the input images
        scale_actual - A scalar value which is the scaling size applied to the batch.
    '''
    # imgs shape is (batch, channel, height, width) and height must equal width
    assert imgs.shape[2]==imgs.shape[3]
    min_scale, max_scale = scale_range
    scale_approximate = torch.rand(1).item() * (max_scale-min_scale) + min_scale    
    hw_original = imgs.shape[2]

    if scale_approximate > 1.0:
        # Zooming in / enlarging
        hw_crop = int(round(hw_original / scale_approximate))
        scale_actual = hw_original / hw_crop
        if scale_actual!=1.0:
            top, left = torch.randint(0, hw_original-hw_crop+1, (2,))
            imgs_scaled = F.resized_crop(imgs, top, left, hw_crop, hw_crop, (hw_original, hw_original), interp_mode)
            return imgs_scaled, scale_actual
        else:
            return imgs, 1.0
    else:
        # Zooming out / shrinking
        hw_shrunk = int(round(hw_original * scale_approximate))
        scale_actual = hw_shrunk / hw_original
        if scale_actual!=1.0:
            imgs_scaled = F.resize(imgs, hw_shrunk, interpolation=interp_mode)
            hw_diff = hw_original-hw_shrunk
            top, left = torch.randint(0, hw_diff+1, (2,))
            right = hw_diff - left
            bottom = hw_diff - top
            imgs_scaled = F.pad(imgs_scaled, (left, top, right, bottom), fill=pad_constant)
            return imgs_scaled, scale_actual
        else:
            return imgs, 1.0


def compute_loss(scale1, scale2, scale_est1, scale_est2):
    # The head must produce an estimate of the scale. However, there is no meaningful
    # units of scale, so the loss function is based on the ratio of the scales for
    # two images that are otherwise identical. Taking the log of the ratio will give
    # ground truth values that are symmetrically centered about zero.
    # What is the proper loss fuction, however? MSE? The distribution is reminiscent
    # of Gaussian, so something specialized for that? Or perhaps something that
    # weights outliers so the model doesn't get stuck just guessing the same scale
    # value for each image?
    bias = 0.0
    ratio_pred = (scale_est1+bias)/(scale_est2+bias)
    ratio_gt = torch.full(scale_est1.shape, (scale1+bias)/(scale2+bias)).cuda()
    loss = torch.nn.functional.mse_loss(ratio_pred, ratio_gt)
    return loss


def MAPELoss(target, output):
  return torch.mean(torch.abs((target - output) / target))


def train(model, head_estimator, optimizer, loader, epoch, n, avgpool):
    head_estimator.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):

        # for i in range(args.batch_size_per_gpu):
        #     x = np.transpose(inp[i,:,:,:].numpy(), (1, 2, 0)) * STD + MEAN
        #     plt.figure(1)
        #     plt.clf()
        #     plt.imshow(x)
        #     plt.title(f'Area: {target[i]}')
        #     # plt.waitforbuttonpress()
        #     import pdb
        #     pdb.set_trace()

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = [x[:, 0] for x in intermediate_output]
                if avgpool:
                    output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
                output = torch.cat(output, dim=-1)
            else:
                output = model(inp)
        area_est = head_estimator(output)[:, 0]

        # Compute loss
        loss = MAPELoss(target/1000., area_est)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged train stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, head_estimator, n, avgpool):
    head_estimator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):

        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        outputs = []
        if "vit" in args.arch:
            intermediate_output = model.get_intermediate_layers(inp, n)
            output = [x[:, 0] for x in intermediate_output]
            if avgpool:
                output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
            output = torch.cat(output, dim=-1)
        else:
            output = model(inp)
        area_est = head_estimator(output)[:, 0]

        # Compute loss
        loss = MAPELoss(target/1000., area_est)

        metric_logger.update(loss=loss.item())

    print("Averaged val stats:  ", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def visualize_predictions(loader, model, head_estimator, n, avgpool):
    head_estimator.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Visualize:'
    print('\nShowing four image pairs per batch. Click on the figure to go to next batch.\n')
    for inp, target in metric_logger.log_every(loader, 1, header):

        inp1, scale1 = scale_batch(inp, (MIN_SCALE, MAX_SCALE))
        inp2, scale2 = scale_batch(inp, (MIN_SCALE, MAX_SCALE))
        # inp1 = torch.clone(inp)
        # inp2 = torch.clone(inp)
        # scale1 = 1.0
        # scale2 = 1.0

        # move to gpu
        inp1 = inp1.cuda(non_blocking=True)
        inp2 = inp2.cuda(non_blocking=True)

        # forward
        outputs = []
        for inp in [inp1, inp2]:
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = [x[:, 0] for x in intermediate_output]
                if avgpool:
                    output.append(torch.mean(intermediate_output[-1][:, 1:], dim=1))
                output = torch.cat(output, dim=-1)
            else:
                output = model(inp)
            outputs.append(output)
        out1, out2 = outputs
        scale_est1 = head_estimator(out1).cpu().numpy()[:,0]
        scale_est2 = head_estimator(out2).cpu().numpy()[:,0]

        # Convert back to numpy and 0.0 to 1.0 colorscale
        inp1 = inp1.cpu().numpy()
        inp2 = inp2.cpu().numpy()
        inp1 = inp1 * np.reshape(STD, (1, 3, 1, 1)) + np.reshape(MEAN, (1, 3, 1, 1))
        inp2 = inp2 * np.reshape(STD, (1, 3, 1, 1)) + np.reshape(MEAN, (1, 3, 1, 1))
        inp1 = np.transpose(inp1, (0, 2, 3, 1))
        inp2 = np.transpose(inp2, (0, 2, 3, 1))
        inp1 = np.clip(inp1, 0, 1)  # values may yet bet out of range due to numerical precision, so clip
        inp2 = np.clip(inp2, 0, 1)  # values may yet bet out of range due to numerical precision, so clip

        # Plot n_pairs images pairs from this batch
        n_pairs = 4
        # n_pairs = 11
        fig = plt.figure(1, figsize=(6, 10))
        plt.clf()
        fig.suptitle(f'scale1={scale1:0.2f}, scale2={scale2:0.2f}, ratio={scale1/scale2:0.2f}')
        for i in range(n_pairs):
            plt.subplot(n_pairs, 2, 2*i+1)
            plt.imshow(inp1[i])
            plt.title(f'pred={scale_est1[i]:0.2f}, ratio={scale_est1[i]/scale_est2[i]:0.2f}')
            plt.axis('off')
            plt.subplot(n_pairs, 2, 2*i+2)
            plt.imshow(inp2[i])
            plt.title(f'pred={scale_est2[i]:0.2f}, ratio={scale_est1[i]/scale_est2[i]:0.2f}')
            plt.axis('off')
        # print(scale_est1)
        plt.waitforbuttonpress()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


class LinearEstimator(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, args):
        super(LinearEstimator, self).__init__()
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softplus': nn.Softplus(),
        }
        self.linear = nn.Linear(dim, 1)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        self.activation = activations[args.output_act]

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # Layer(s)
        return self.activation(self.linear(x))


class MlpEstimator(nn.Module):
    """MLP to train on top of frozen features"""
    def __init__(self, dim, args):
        super(MlpEstimator, self).__init__()
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'softplus': nn.Softplus(),
        }
        self.hidden = [nn.Linear(dim, args.n_hidden_nodes)]
        self.hidden[0].weight.data.normal_(mean=0.0, std=0.01)
        self.hidden[0].bias.data.zero_()

        for i in range(1, args.n_hidden_layers):
            self.hidden.append(nn.Linear(args.n_hidden_nodes, args.n_hidden_nodes))
            self.hidden[-1].weight.data.normal_(mean=0.0, std=0.01)
            self.hidden[-1].bias.data.zero_()

        self.hidden = nn.ModuleList(self.hidden)  # Need so all layers will be moved to GPU when calling model.cuda()

        self.output = nn.Linear(args.n_hidden_nodes, 1)
        self.output.weight.data.normal_(mean=0.0, std=0.01)
        self.output.bias.data.zero_()

        self.output_act = activations[args.output_act]
        self.hidden_act = activations[args.hidden_act]

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # Layer(s)
        for i in range(args.n_hidden_layers):
            x = self.hidden_act(self.hidden[i](x))
        return self.output_act(self.output(x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')

    parser.add_argument('--head_type', default='linear', type=str, help='Linear or mlp')
    parser.add_argument('--output_act', default='softplus', type=str, help='Activation of head model final layer')
    parser.add_argument('--hidden_act', default='relu', type=str, help='Activation of head model hidden layers')
    parser.add_argument('--n_hidden_layers', default=1, type=int, help='Number of hidden layers, if using MLP head model')
    parser.add_argument('--n_hidden_nodes', default=40, type=int, help='Number of nodes per hidden layers, if using MLP head model')

    parser.add_argument('--view', default=False, type=utils.bool_flag, help='Load trained linear head and visualize results')
    parser.add_argument('--view_dataset', default='val', type=str, help='If visualizing results, specifies whether to use "train" or "va" dataset')
    args = parser.parse_args()
    eval_linear(args)



# Command to visualize validation data:
# python eval_linear_scale.py --data_path /Data/DairyTech/Flickr_cows_train_val_sets/ --num_workers 8 --view True


# python eval_linear_cale.py --data_path /Data/DairyTech/scaled_sets/test_sample/ --num_workers 8 --view True


# python eval_linear_area.py \
# --arch vit_small \
# --n_last_blocks 4 \
# --patch_size 16 \
# --avgpool_patchtokens False \
# --batch_size_per_gpu 73 \
# --epochs 500 \
# --num_workers 8 \
# --head_type mlp \
# --output_act softplus \
# --hidden_act relu \
# --n_hidden_layers 1 \
# --n_hidden_nodes 40 \
# --output_dir ./area_head_temp \
# 2>/dev/null


# python eval_linear_scale.py \
# --arch vit_base \
# --n_last_blocks 1 \
# --patch_size 8 \
# --avgpool_patchtokens True \
# --batch_size_per_gpu 32 \
# --data_path /Data/DairyTech/Flickr_cows_train_val_sets/ \
# --num_workers 8 \
# --output_dir ./scale_head_base_linear \
# --view True


# python eval_linear_scale.py \
# --arch vit_small \
# --n_last_blocks 4 \
# --patch_size 16 \
# --avgpool_patchtokens False \
# --batch_size_per_gpu 128 \
# --data_path /Data/DairyTech/Flickr_cows_train_val_sets/ \
# --num_workers 8 \
# --head_type mlp \
# --output_act softplus \
# --hidden_act relu \
# --n_hidden_layers 1 \
# --n_hidden_nodes 40 \
# --output_dir ./scale_head_small_mlp_L1_N40 \
# 2>/dev/null


# python eval_linear_scale.py \
# --arch vit_small \
# --n_last_blocks 4 \
# --patch_size 16 \
# --avgpool_patchtokens False \
# --batch_size_per_gpu 32 \
# --data_path /Data/DairyTech/Flickr_cows_train_val_sets/ \
# --num_workers 8 \
# --head_type linear \
# --output_act softplus \
# --hidden_act relu \
# --n_hidden_layers 1 \
# --n_hidden_nodes 40 \
# --output_dir ./temp \


