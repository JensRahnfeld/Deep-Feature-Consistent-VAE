import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


vggnet = models.__dict__['vgg19'](pretrained=True)
for param in vggnet.features.parameters():
    param.requires_grad = False


class VGG123(nn.Module):
    def __init__(self):
        super(VGG123, self).__init__()
        self.conv1_1 = vggnet.features[0]
        self.conv1_2 = vggnet.features[2]
        self.conv2_1 = vggnet.features[5]
        self.conv2_2 = vggnet.features[7]
        self.conv3_1 = vggnet.features[10]

        self.max1 = vggnet.features[4]
        self.max2 = vggnet.features[9]
    
    def forward(self, x):
        out1 = F.relu(self.conv1_1(x), inplace=True)
        x = F.relu(self.conv1_2(out1), inplace=True)
        x = self.max1(x)
        
        out2 = F.relu(self.conv2_1(x), inplace=True)
        x = F.relu(self.conv2_2(out2), inplace=True)
        x = self.max2(x)

        out3 = F.relu(self.conv3_1(x), inplace=True)

        return out1, out2, out3


class VGG345(nn.Module):
    def __init__(self):
        super(VGG345, self).__init__()
        self.conv1_1 = vggnet.features[0]
        self.conv1_2 = vggnet.features[2]
        self.conv2_1 = vggnet.features[5]
        self.conv2_2 = vggnet.features[7]
        self.conv3_1 = vggnet.features[10]
        self.conv3_2 = vggnet.features[12]
        self.conv3_3 = vggnet.features[14]
        self.conv3_4 = vggnet.features[16]
        self.conv4_1 = vggnet.features[19]
        self.conv4_2 = vggnet.features[21]
        self.conv4_3 = vggnet.features[23]
        self.conv4_4 = vggnet.features[25]
        self.conv5_1 = vggnet.features[28]

        self.max1 = vggnet.features[4]
        self.max2 = vggnet.features[9]
        self.max3 = vggnet.features[18]
        self.max4 = vggnet.features[27]

    def forward(self, x):
        x = F.relu(self.conv1_1(x), inplace=True)
        x = F.relu(self.conv1_2(x), inplace=True)
        x = self.max1(x)
        
        x = F.relu(self.conv2_1(x), inplace=True)
        x = F.relu(self.conv2_2(x), inplace=True)
        x = self.max2(x)

        out1 = F.relu(self.conv3_1(x), inplace=True)
        x = F.relu(self.conv3_2(out1), inplace=True)
        x = F.relu(self.conv3_3(x), inplace=True)
        x = F.relu(self.conv3_4(x), inplace=True)
        x = self.max3(x)

        out2 = F.relu(self.conv4_1(x), inplace=True)
        x = F.relu(self.conv4_2(out2), inplace=True)
        x = F.relu(self.conv4_3(x), inplace=True)
        x = F.relu(self.conv4_4(x), inplace=True)
        x = self.max4(x)

        out3 = F.relu(self.conv5_1(x), inplace=True)

        return out1, out2, out3