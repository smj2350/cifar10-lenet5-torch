import torch
import torch.nn.functional as F
import torch.nn as nn

class MAPLoss(nn.Module):
    def __init__(self, num_classes, prior=None):
        super(MAPLoss, self).__init__()
        # 사전 확률 설정 (균등 분포 또는 사용자 정의 분포)
        if prior is None:
            self.prior = torch.ones(num_classes) / num_classes  # 균등 분포
        else:
            self.prior = torch.tensor(prior)

    def forward(self, outputs, targets):
        # log_softmax를 통해 각 클래스의 로그 확률 계산
        log_probs = F.log_softmax(outputs, dim=1)

        # 사전 확률 적용: MAP = log(p(y|x)) + log(p(y))
        log_prior = torch.log(self.prior).to(outputs.device)
        log_map = log_probs + log_prior

        # NLLLoss를 기반으로 손실 계산
        loss = F.nll_loss(log_map, targets)
        return loss
