import torch
import torch.nn as nn
import torch.nn.functional as F


def num_flat_features(x):
    num_features = 1
    
    for dim in x.size()[1:]:
        num_features *= dim
    
    return num_features


class _Base(nn.Module):
    def __init__(self, dim_latent=100):
        super(_Base, self).__init__()

        self.dim_latent = dim_latent
    
    def forward(self, x):
        raise NotImplementedError()


class Encoder(_Base):
    def __init__(self, *args, **kwargs):
        super(Encoder, self).__init__(*args, **kwargs)

        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)

        self.fc_mu = nn.Linear(256 * 4 * 4, self.dim_latent)
        self.fc_logvar = nn.Linear(256 * 4 * 4, self.dim_latent)
    
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.bn4(self.conv4(x)))

        x = x.view(-1, num_flat_features(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar
    

class Decoder(_Base):
    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

        self.fc1 = nn.Linear(self.dim_latent, 256 * 4 * 4)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample4 = nn.Upsample(scale_factor=2, mode='nearest')

        self.pad1 = nn.ReplicationPad2d(1)
        self.pad2 = nn.ReplicationPad2d(1)
        self.pad3 = nn.ReplicationPad2d(1)
        self.pad4 = nn.ReplicationPad2d(1)

        self.conv1 = nn.Conv2d(256, 128, 3, 1)
        self.conv2 = nn.Conv2d(128, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 32, 3, 1)
        self.conv4 = nn.Conv2d(32, 3, 3, 1)

        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, 256, 4, 4)

        x = F.leaky_relu(self.bn1(self.conv1(self.pad1(self.upsample1(x)))))
        x = F.leaky_relu(self.bn2(self.conv2(self.pad2(self.upsample2(x)))))
        x = F.leaky_relu(self.bn3(self.conv3(self.pad3(self.upsample3(x)))))
        x = self.conv4(self.pad4(self.upsample4(x)))

        return x


class VAE(_Base):
    def __init__(self, *args, **kwargs):
        super(VAE, self).__init__(*args, **kwargs)

        self.encoder = Encoder(self.dim_latent)
        self.decoder = Decoder(self.dim_latent)
    
    def sample(self, mu, stdev):
        eps = torch.randn(stdev.shape, device=stdev.device)
        latent = mu + stdev * eps

        return latent
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        stdev = torch.exp(logvar / 2.0)

        latent = self.sample(mu, stdev)

        x_rec = self.decoder(latent)

        return x_rec, mu, logvar