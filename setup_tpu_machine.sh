# tensorflow can mess up the connection with the TPU (in some cases)
sudo pip uninstall tensorflow

# install jax-smi to show the TPU usage
#sudo pip install jax-smi # this somehow did not work

# install the requirements
sudo pip install -r requirements.txt

# install git lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# config git
git config --global credential.helper store

# login to huggingface
huggingface-cli login

# login to wandb
wandb login




