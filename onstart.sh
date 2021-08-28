## Download and untar files from AWS S3

# Some convenient aliases
alias lh='ls -lGFhp'
alias smi='nvidia-smi'

# Install common and necessary apt packages
apt update
apt -y install vim
apt -y install curl
apt -y install unzip

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
echo aws_access_key_id=AKIAZF5DIH3HQQTJ5AN4 >> credentials
echo aws_secret_access_key=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 >> credentials

# Test AWS credentials with ls
# aws s3 ls s3://bicog-datasets/

# Download the tar file from AWS
cd /workspace/Downloads
# aws s3 cp s3://bicog-datasets/ilsvrc2012/ILSVRC2012_img_val.tar ./
aws s3 cp s3://bicog-datasets/cows/flickr_cows_postprocessed.tar ./

# Delete the onstart.sh file to remove AWS credentials?
# rm /root/onstart.sh


# Untar images
tar -xzvf flickr_cows_postprocessed.tar


# Install apt and pip packages needed for DINO
pip install torch
pip install timm


# Download DINO code
cd /workspace
git clone git@github.com:mattroos/dino.git


# Launch training
cd /workspace/dino
python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch vit_small --batch_size_per_gpu 24 --data_path /workspace/Downloads/flicker_cows_postprocessed --output_dir ./output

