## Set up remote machine for training of ViT and head models, using dino


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
echo aws_access_key_id=KEY >> credentials
echo aws_secret_access_key=KEY >> credentials


# Test AWS credentials with ls
# aws s3 ls s3://bicog-datasets/

# Download the tar file from AWS
cd /workspace/Downloads
aws s3 cp s3://bicog-datasets/cows/Flickr_fields_train_val_sets.tar ./
aws s3 cp s3://bicog-datasets/cows/scaled_segmented_train_val_sets.tar ./


# Delete the onstart.sh file to remove AWS credentials?
# rm /root/onstart.sh


# Untar images
tar -xvf Flickr_fields_train_val_sets.tar
tar -xvf scaled_segmented_train_val_sets.tar


# Install apt and pip packages needed for DINO
pip install torch
pip install timm
pip install matplotlib


# Store github ssh public key for vast.ai
echo PUBLIC_KEY >> /root/.ssh/id_ed25519.pub

# Store github ssh private key, and add to ssh-agent
echo PRIVATE_KEY >> /root/.ssh/id_ed25519
chmod 400 /root/.ssh/id_ed25519

# Start the ssh-agent in the background.
eval "$(ssh-agent -s)"

# Add your SSH private key to the ssh-agent.
ssh-add /root/.ssh/id_ed25519


# Download DINO code
cd /workspace
git clone git@github.com:mattroos/dino.git

