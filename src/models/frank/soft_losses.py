import torch
import torch.nn.functional as F


class SoftBCEWithLogitsLoss:
    def __init__(self, temperature=1):
        self.temperature = temperature

    def __call__(self, input, target):
        input, target = input / self.temperature, target / self.temperature
        target = torch.sigmoid(target)
        return F.binary_cross_entropy_with_logits(input, target)


class SoftCrossEntropyLoss:
    def __init__(self, temperature=1):
        self.temperature = temperature

    def __call__(self, input, target):
        """
        :param inputs: predictions
        :param target: target labels
        :return: loss
        """
        input, target = input / self.temperature, target / self.temperature

        target = F.softmax(target, dim=1)
        log_likelihood = - F.log_softmax(input, dim=1)
        sample_num, class_num = target.shape
        loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num

        return loss
