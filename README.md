## Deep-Feature-Consistent-VAE
Pytorch Implementation of Hou, Shen, Sun, Qiu, "Deep Feature Consistent Variational Autoencoder", 2016

# Requirements
* Python 3.6.9
* numpy 1.19.0
* pillow 7.2.0
* pytorch 1.5.1
* tensorboard 2.2.2
* torchvision 0.6.1

# 1) Installing Dependencies
```
pip3 install -r requirements.txt
```

# 2) Train Vae
```
python3 train_vae.py --imgdir <path> --workers 8 -o vae
```