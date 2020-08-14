import torch
import torch.nn.functional as F
from models.vggnet import VGG123, VGG345

vgg123 = VGG123()
vgg123 = vgg123.eval().cuda()
vgg345 = VGG345()
vgg345 = vgg345.eval().cuda()


def kl_loss(mu, logvar, beta):
    loss = -0.5 * torch.sum(-torch.exp(logvar/2.0) - mu.pow(2) + 1 + logvar, dim=[1])
    loss = beta * torch.mean(loss)

    return loss


def vgg123_loss(x_rec, x_true, alpha):
    out1_rec, out2_rec, out3_rec = vgg123(x_rec)
    out1_true, out2_true, out3_true = vgg123(x_true)

    loss1 = F.mse_loss(out1_rec, out1_true)
    loss2 = F.mse_loss(out2_rec, out2_true)
    loss3 = F.mse_loss(out3_rec, out3_true)
    loss = alpha * 0.5 * (loss1 + loss2 + loss3)

    return loss


def vgg345_loss(x_rec, x_true, alpha):
    out1_rec, out2_rec, out3_rec = vgg345(x_rec)
    out1_true, out2_true, out3_true = vgg345(x_true)

    loss1 = F.mse_loss(out1_rec, out1_true)
    loss2 = F.mse_loss(out2_rec, out2_true)
    loss3 = F.mse_loss(out3_rec, out3_true)
    loss = alpha * 0.5 * (loss1 + loss2 + loss3)

    return loss