from model.self_supervision import ContextualInference
from model.self_supervision import NeighborPredict
import torch
from torch import nn


class SelfSupervision(nn.Module):

    def __init__(self):
        super(SelfSupervision, self).__init__()
