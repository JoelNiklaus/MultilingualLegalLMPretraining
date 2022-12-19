# tensorflow can mess up the connection with the TPU (in some cases)
sudo pip uninstall tensorflow -y

# install jax-smi to show the TPU usage
#sudo pip install jax-smi # this somehow did not work

# install the requirements
sudo pip install -r requirements.txt

# install git lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

# config git
sudo git config --global credential.helper store

# prevent strange error: fatal: detected dubious ownership in repository at '/home/joelniklaus/MultilingualLegalLMPretraining'
sudo git config --global --add safe.directory /home/joelniklaus/MultilingualLegalLMPretraining

# login to huggingface
sudo huggingface-cli login

# login to wandb
sudo wandb login

# set .bashrc
echo 'cd MultilingualLegalLMPretraining/' >> ../.bashrc



