import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


vggnet = models.__dict__['vgg19'](pretrained=True)
for param in vggnet.features.parameters():
    param.requires_grad = False


class VGG123(nn.Module):
    def __init__(self):
        super(VGG123, self).__init__()
        self.conv1 = vggnet.features[0]
        self.conv2 = vggnet.features[2]
        self.conv3 = vggnet.features[5]

        self.max1 = vggnet.features[4]
    
    def forward(self, x):
        out1 = F.relu(self.conv1(x), inplace=True)
        out2 = F.relu(self.conv2(out1), inplace=True)
        out3 = F.relu(self.conv3(self.max1(out2)), inplace=True)

        return out1, out2, out3


class VGG345(nn.Module):
    def __init__(self):
        super(VGG345, self).__init__()
        self.conv1 = vggnet.features[0]
        self.conv2 = vggnet.features[2]
        self.conv3 = vggnet.features[5]
        self.conv4 = vggnet.features[7]
        self.conv5 = vggnet.features[10]

        self.max1 = vggnet.features[4]
        self.max2 = vggnet.features[9]

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        out1 = F.relu(self.conv3(self.max1(x)), inplace=True)
        out2 = F.relu(self.conv4(out1), inplace=True)
        out3 = F.relu(self.conv5(self.max2(out2)), inplace=True)

        return out1, out2, out3