

# parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
# parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
#     for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
# parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
#     help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
#     We typically set this to False for ViT-Small and to True with ViT-Base.""")
# parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
# parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
# parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
# parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
# parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
# parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
#     training (highest LR used during training). The learning rate is linearly scaled
#     with the batch size, and specified here for a reference batch size of 256.
#     We recommend tweaking the LR depending on the checkpoint evaluated.""")
# parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
# parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
#     distributed training; see https://pytorch.org/docs/stable/distributed.html""")
# parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
# parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
# parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
# parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
# parser.add_argument('--output_dir', default=".", help='Path to save logs and checkpoints')
# parser.add_argument('--num_labels', default=1000, type=int, help='Number of labels for linear classifier')
# parser.add_argument('--view', default=False, type=utils.bool_flag, help='Load trained linear head and visualize results')
# parser.add_argument('--view_dataset', default='val', type=str, help='If visualizing results, specifies whether to use "train" or "va" dataset')


# Current number of training samples is 7544. Try to choose batch size such that it divides evenly?
# Current number of test samples is 838. Try to choose batch size such that it divides evenly?


## Train head on top of ViT-small
python eval_linear_scale.py \
--arch vit_small \
--n_last_blocks 4 \
--patch_size 16 \
--avgpool_patchtokens False \
--batch_size_per_gpu 128 \
--data_path /Data/DairyTech/Flickr_cows_train_val_sets/ \
--num_workers 8 \
--output_dir ./scale_head_small_linear \
2>/dev/null


## Train head on top of ViT-base
python eval_linear_scale.py \
--arch vit_base \
--n_last_blocks 1 \
--patch_size 8 \
--avgpool_patchtokens True \
--batch_size_per_gpu 64 \
--data_path /Data/DairyTech/Flickr_cows_train_val_sets/ \
--num_workers 8 \
--output_dir ./scale_head_base_linear \
2>/dev/null


## Visualize scaled images, with ground truth and prediction scale values
python eval_linear_scale.py \
--data_path /Data/DairyTech/Flickr_cows_train_val_sets/ \
--num_workers 8 \
--view True




