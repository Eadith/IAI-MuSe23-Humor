import torch
import torch.nn as nn


class CCCLoss(nn.Module):
    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, y_pred, y_true, seq_lens=None):
        if seq_lens is not None:
            mask = torch.ones_like(y_true, device=y_true.device)
            for i, seq_len in enumerate(seq_lens):
                mask[i, seq_len:] = 0
        else:
            mask = torch.ones_like(y_true, device=y_true.device)
        y_true_mean = torch.sum(y_true * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)
        y_pred_mean = torch.sum(y_pred * mask, dim=1, keepdim=True) / torch.sum(mask, dim=1, keepdim=True)

        y_true_var = torch.sum(mask * (y_true - y_true_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                    keepdim=True)
        y_pred_var = torch.sum(mask * (y_pred - y_pred_mean) ** 2, dim=1, keepdim=True) / torch.sum(mask, dim=1,
                                                                                                    keepdim=True)

        cov = torch.sum(mask * (y_true - y_true_mean) * (y_pred - y_pred_mean), dim=1, keepdim=True) \
              / torch.sum(mask, dim=1, keepdim=True)

        ccc = torch.mean(2.0 * cov / (y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2), dim=0)
        ccc = ccc.squeeze(0)
        ccc_loss = 1.0 - ccc

        return ccc_loss


# wraps BCEWithLogitsLoss, but constructor accepts (and ignores) argument seq_lens
class BCEWithLogitsLossWrapper(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20))

    def forward(self, y_pred, y_true, seq_lens=None):
        return self.loss_fn(y_pred, y_true)


# wraps MSELoss, but constructor accepts (and ignores) argument seq_lens
class MSELossWrapper(nn.Module):

    def __init__(self, reduction):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, y_pred, y_true, seq_lens=None):
        return self.loss_fn(y_pred, y_true)


# wraps BCELoss, but constructor accepts (and ignores) argument seq_lens
class BCELossWrapper(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCELoss()      

    def forward(self, y_pred, y_true, seq_lens=None):
        return self.loss_fn(y_pred, y_true)


def get_segment_wise_labels(labels):
    # collapse labels to one label per segment (as originally for MuSe-Sent)
    segment_labels = []
    for i in range(labels.size(0)):
        segment_labels.append(labels[i, 0, :])
    labels = torch.stack(segment_labels).long()
    return labels


def get_segment_wise_logits(logits, feature_lens):
    # determines exactly one output for each segment (by taking the last timestamp of each segment)
    segment_logits = []
    for i in range(logits.size(0)):
        segment_logits.append(logits[i, feature_lens[i] - 1, :])  # (batch-size, frames, classes)
    logits = torch.stack(segment_logits, dim=0)
    return logits


class Focal_Loss(nn.Module):

	def __init__(self, alpha=0.8125, gamma=0.5): #gamma=0.5, alpha=0.8125
		super(Focal_Loss,self).__init__()
		self.alpha=alpha
		self.gamma=gamma
	
	def forward(self, preds, labels, seq_lens=None):
		""" preds经过sigmoid后的概率值 """
		eps=1e-7
		loss_1=-1*self.alpha*torch.pow((1-preds),self.gamma)*torch.log(preds+eps)*labels
		loss_0=-1*(1-self.alpha)*torch.pow(preds,self.gamma)*torch.log(1-preds+eps)*(1-labels)
		loss=loss_0+loss_1
		return torch.mean(loss)


