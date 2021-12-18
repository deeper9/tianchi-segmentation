from torch import nn

class DiceScores(nn.Module):
    def __init__(self, eps=1e-3, threshold=0.5):
        super(DiceScores, self).__init__()
        self.threshold = threshold
        self.eps = eps

    def forward(self, y_pred, y_true):
        p = y_pred.reshape(-1)
        t = y_true.reshape(-1)

        p = p > self.threshold
        t = t > self.threshold
        uion = p.sum() + t.sum()

        overlap = (p * t).sum()
        dice = 2 * overlap / (uion + self.eps)
        return dice
