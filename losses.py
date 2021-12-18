from torch import nn
import torch

class DiceLoss(nn.Module):
    def __init__(self, smooth=1., dims=(-2, -1)):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims

    def forward(self, predict, mask_true):
        tp = (predict * mask_true).sum(self.dims)
        fp = (predict * (1 - mask_true)).sum(self.dims)
        fn = ((1 - predict) * mask_true).sum(self.dims)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()
        return 1 - dc


class EvalLoss(nn.Module):
    def __init__(self, ratio=0.8, hard=False):
        super(EvalLoss, self).__init__()
        self.ratio = ratio
        self.hard = hard
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, predict, mask_true):
        bce = self.bce(predict, mask_true)
        if self.hard:
            dice = self.dice_loss((predict.sigmoid()).float() > 0.5, mask_true)
        else:
            dice = self.dice_loss(predict.sigmoid(), mask_true)
        return self.ratio * bce + (1 - self.ratio) * dice


if __name__ == '__main__':
    f2 = EvalLoss()
    inputs = torch.randn(size=(4, 1, 3, 3))
    targets = torch.randn(size=(4, 1, 3, 3))
    d_loss = f2(inputs, targets)
    print(d_loss)
