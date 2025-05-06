import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self,alpha=0.25, gamma=2.0,reduction='mean'):
        super(FocalLoss,self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs : [batch_size], raw logits
        # targets : [batch_size], binary  labels
        BCE_loss = F.binary_cross_entropy_with_logits(inputs,targets.float(), reduction='none')
        probs = torch.sigmoid(inputs) # 확률값
        p_t = probs * targets + (1-probs) * (1 - targets) # p_t = p if y=1 else 1-p
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
class FocalLossMultiClass(nn.Module):
    def __init__(self,alpha=None,gamma=2.0,reduction='mean'):
        super(FocalLossMultiClass,self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha # list or tensor of class weights

    def forward(self,inputs,targets):
        log_probs = F.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        targets_onehot = F.one_hot(targets,num_classes=inputs.size(1)).float()

        pt = (probs * targets_onehot).sum(dim=1) # p_t = p if y=1 else 1-p
        log_pt = (log_probs * targets_onehot).sum(dim=1) # log(p_t)
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            loss = -alpha_t * (1-pt) ** self.gamma * log_pt
        else:
            loss = -(1-pt) ** self.gamma * log_pt

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss            