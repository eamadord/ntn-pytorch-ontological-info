import torch as t
import torch.nn as nn

class Tensor_Loss(t.nn.Module):
    def __init__(self):
        super(Tensor_Loss, self).__init__()

    def forward(self, predictions, regularization, parameters):
        scalar = t.Tensor([0])
        aux = 1-predictions[0]+predictions[1]
        tmp1 = t.max(scalar.expand_as(aux), aux.data)
        tmp1 = t.sum(tmp1)
        tmp2 = t.sqrt(sum([t.sum(var**2) for var in parameters]))
        loss = tmp1 + (regularization * tmp2)
        return loss