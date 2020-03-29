from pathlib import Path

import torch as T
import torch.nn as nn
import torch.nn.functional as F

from agent.config import Config
from agent.state.model import StateEncoder


class TransitionModel(nn.Module):

    def __init__(self, state_size: int, reg_coef: float, prune_proba: float,
                 prune_epsilon: float):
        super().__init__()
        self._fc1 = nn.Linear(state_size, state_size)
        self._ln1 = nn.LayerNorm(state_size, elementwise_affine=True)
        self._fc2 = nn.Linear(state_size, state_size)

        self._reg_coef = reg_coef
        self._prune_proba = prune_proba
        self._prune_epsilon = prune_epsilon

        self._mask_by_param = []
        with T.no_grad():
            for i, p in enumerate(self.parameters()):
                filt = T.rand_like(p.data)
                p.data[filt < prune_proba] = 0.

                mask = T.zeros_like(filt)
                mask[filt >= prune_proba] = 1
                self._mask_by_param.append(mask)

    def l1_loss(self):
        sum_l1 = sum(T.norm(p, 1) for p in self.parameters()
                     if p.requires_grad)
        return self._reg_coef * sum_l1 / self.total_params

    def forward(self, input_):
        x = F.relu(self._ln1(self._fc1(input_)), inplace=True)
        x = self._fc2(x)
        return x

    def prune_weights(self):
        with T.no_grad():
            e = self._prune_epsilon
            for p in self.parameters():
                p.data[T.abs(p.data) < e] = 0.

    def mask_grads(self):
        with T.no_grad():
            for i, p in enumerate(self.parameters()):
                p.data[self._mask_by_param[i] == 0] = 0.


def train_transition(conf: Config, device: T.device, encoder: StateEncoder,
                     out_dir: Path):
    encoder.eval()
    models = []
    for _ in range(conf.trans.num_models):
        m = TransitionModel(state_size=encoder.out_size,
                            reg_coef=conf.trans.reg_coef,
                            prune_proba=conf.trans.prune_proba,
                            prune_epsilon=conf.trans.prune_epsilon)
        models.append(m)
