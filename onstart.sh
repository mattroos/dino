## Download and untar files from AWS S3


# Add some convenient aliases to .bashrc
echo "alias lh='ls -lGFhp'" >> ~/.bashrc
echo "alias smi='nvidia-smi'" >> ~/.bashrc

# Install common and necessary apt packages
apt update
apt -y install vim
apt -y install curl
apt -y install unzip
apt -y install git

# Install AWS CLI
mkdir Downloads
cd Downloads
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Set up AWS credentials
cd ~
mkdir .aws
cd .aws
echo [default] >> credentials
echo aws_access_key_id=KEY_ID >> credentials
echo aws_secret_access_key=SECRET_KEY >> credentials


# Test AWS credentials with ls
# aws s3 ls s3://bicog-datasets/

# Download the tar file from AWS
cd /workspace/Downloads
# aws s3 cp s3://bicog-datasets/ilsvrc2012/ILSVRC2012_img_val.tar ./
# aws s3 cp s3://bicog-datasets/cows/flickr_cows_postprocessed.tar ./
aws s3 cp s3://bicog-datasets/cows/Flickr_cows_train_val_sets.tar ./


# Delete the onstart.sh file to remove AWS credentials?
# rm /root/onstart.sh


# Untar images
# tar -xzvf Flickr_cows_train_val_sets.tar.gz
tar -xvf Flickr_cows_train_val_sets.tar


# Install apt and pip packages needed for DINO
pip install torch
pip install timm
pip install matplotlib


# Store github ssh public key for vast.ai
echo SSH_KEY >> /root/.ssh/id_ed25519.pub


# Download DINO code
cd /workspace
git clone git@github.com:mattroos/dino.git


# Launch training
cd /workspace/dino

# ## Train transformer using SSL
# python -m torch.distributed.launch --nproc_per_node=4 main_dino.py \
# --arch vit_small \
# --batch_size_per_gpu 24 \
# --data_path /workspace/Downloads/flicker_cows_postprocessed \
# --output_dir ./output


# ## Train head on top of ViT-small
# python -m torch.distributed.launch --nproc_per_node=4 eval_linear_scale.py \
# --arch vit_small \
# --n_last_blocks 4 \
# --patch_size 16 \
# --avgpool_patchtokens False \
# --batch_size_per_gpu 128 \
# --data_path /workspace/Downloads/Flickr_cows_train_val_sets/ \
# --num_workers 8 \
# --output_dir ./scale_head_small_linear \
# 2>/dev/null

# # Tar and copy results to AWS S3
# tar -cvf scale_head_small_linear.tar /workspace/dino/scale_head_small_linear
# aws s3 cp scale_head_small_linear.tar s3://bicog-datasets/cows/scale_head_small_linear.tar


# ## Train head on top of ViT-base
# python -m torch.distributed.launch --nproc_per_node=4 eval_linear_scale.py \
# --arch vit_base \
# --n_last_blocks 1 \
# --patch_size 8 \
# --avgpool_patchtokens True \
# --batch_size_per_gpu 128 \
# --data_path /workspace/Downloads/Flickr_cows_train_val_sets/ \
# --num_workers 8 \
# --output_dir ./scale_head_base_linear \
# 2>/dev/null

# # Tar and copy results to AWS S3
# tar -cvf scale_head_base_linear.tar /workspace/dino/scale_head_base_linear
# aws s3 cp scale_head_base_linear.tar s3://bicog-datasets/cows/scale_head_base_linear.tar


## Train MLP head on top of ViT-small
python -m torch.distributed.launch --nproc_per_node=4 eval_linear_scale.py \
--arch vit_small \
--n_last_blocks 4 \
--patch_size 16 \
--avgpool_patchtokens False \
--batch_size_per_gpu 256 \
--data_path /workspace/Downloads/Flickr_cows_train_val_sets/ \
--num_workers 8 \
--head_type mlp \
--output_act softplus \
--hidden_act relu \
--n_hidden_layers 1 \
--n_hidden_nodes 40 \
--output_dir ./scale_head_small_mlp_L1_N40 \
2>/dev/null

# Tar and copy results to AWS S3
tar -cvf scale_head_small_mlp_L1_N40.tar /workspace/dino/scale_head_small_mlp_L1_N40
aws s3 cp scale_head_small_mlp_L1_N40.tar s3://bicog-datasets/cows/scale_head_small_mlp_L1_N40.tar


## Train MLP head on top of ViT-base
python -m torch.distributed.launch --nproc_per_node=4 eval_linear_scale.py \
--arch vit_base \
--n_last_blocks 1 \
--patch_size 8 \
--avgpool_patchtokens True \
--batch_size_per_gpu 128 \
--data_path /workspace/Downloads/Flickr_cows_train_val_sets/ \
--num_workers 8 \
--head_type mlp \
--output_act softplus \
--hidden_act relu \
--n_hidden_layers 1 \
--n_hidden_nodes 40 \
--output_dir ./scale_head_base_mlp_L1_N40 \
2>/dev/null

# Tar and copy results to AWS S3
tar -cvf scale_head_base_mlp_L1_N40.tar /workspace/dino/scale_head_base_mlp_L1_N40
aws s3 cp scale_head_base_mlp_L1_N40.tar s3://bicog-datasets/cows/scale_head_base_mlp_L1_N40.tar

