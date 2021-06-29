from torch.nn import functional
import torch


def _make_weights_dict(instruments, weights):
    result = dict()
    for name, weight in zip(instruments, weights):
        result[name] = weight
    return result

def si_snr(source, estimate_source, eps=1e-5):
    source = source.squeeze(-1)
    estimate_source = estimate_source.squeeze(-1)
    B,T = source.size()
    source_energy = torch.sum(source ** 2, dim=1).view(B, 1)  # B , 1
    dot = torch.matmul(estimate_source, source.t())  # B , B
    s_target = torch.matmul(dot, source) / (source_energy + eps)  # B , T
    e_noise = estimate_source - source
    snr = 10 * torch.log10(torch.sum(s_target ** 2, dim=1) / (torch.sum(e_noise ** 2, dim=1) + eps) + eps)  # B , 1
    lo = 0 - torch.mean(snr)
    return lo

class MultiLoss:
    def __init__(self, instruments, weights,loss_type):
        super(MultiLoss, self).__init__()
        self.weights = _make_weights_dict(instruments, weights)
        self.loss_type = loss_type

    def __call__(self, predict, target):
        loss, sub_loss = 0, dict()

        for key in self.weights:
            weight = self.weights[key]   

            if self.loss_type == "l2":
                
                cur_loss = weight * functional.mse_loss(predict[key], target[key])

            elif self.loss_type == "l1":
                cur_loss = weight * functional.l1_loss(predict[key], target[key])

            elif self.loss_type == "smooth_l1":   
                cur_loss = weight * functional.smooth_l1_loss(predict[key], target[key])

            elif self.loss_type == "si_snr":   
                cur_loss = weight * si_snr(predict[key], target[key])
                        
            loss = loss + cur_loss
            sub_loss[key] = cur_loss

        return loss, sub_loss
