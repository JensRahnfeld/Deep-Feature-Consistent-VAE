import torch
import torch.nn.functional as F
from models.vggnet import VGG123, VGG345

vgg123 = VGG123().eval()
if torch.cuda.is_available(): vgg123 = vgg123.cuda()
vgg345 = VGG345().eval()
if torch.cuda.is_available(): vgg345 = vgg345.cuda()


def kl_loss(mu, logvar):
    loss = -0.5 * torch.sum(-torch.exp(logvar/2.0) - mu.pow(2) + 1 + (logvar/2.0), dim=[1])
    loss = torch.mean(loss)

    return loss

def l2_loss(x_rec, x_true):
    loss = torch.sum((x_rec - x_true).pow(2), dim=[1,2,3])
    loss = torch.mean(loss)

    return loss

def layer_loss(pred, target):
    C, H, W = pred.size()[1:]
    loss = torch.sum((pred - target).pow(2), dim=[1,2,3]) / (2 * C * H * W)

    return loss

def vgg123_loss(x_rec, x_true):
    out1_rec, out2_rec, out3_rec = vgg123(x_rec)
    out1_true, out2_true, out3_true = vgg123(x_true)

    loss1 = layer_loss(out1_rec, out1_true)
    loss2 = layer_loss(out2_rec, out2_true)
    loss3 = layer_loss(out3_rec, out3_true)
    loss = torch.mean(loss1 + loss2 + loss3)

    return loss

def vgg345_loss(x_rec, x_true):
    out1_rec, out2_rec, out3_rec = vgg345(x_rec)
    out1_true, out2_true, out3_true = vgg345(x_true)

    loss1 = layer_loss(out1_rec, out1_true)
    loss2 = layer_loss(out2_rec, out2_true)
    loss3 = layer_loss(out3_rec, out3_true)
    loss = torch.mean(loss1 + loss2 + loss3)

    return loss