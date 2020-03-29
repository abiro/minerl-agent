from pathlib import Path
from typing import Dict, NamedTuple, Optional
import os

import oyaml as yaml


class TransConfig(NamedTuple):
    """Transition model config"""
    num_models: int
    prune_proba: float
    prune_epsilon: float
    reg_coef: float


class StateConfig(NamedTuple):
    """State model config"""
    batch_size: int
    checkpoint_freq: int
    cnn_out: int
    critic_clamp: float
    critic_iters: int
    critic_turbo_iters: int
    epochs: int
    frame_gen_filters: int
    lr: float
    inv_out: int
    num_act: int
    num_inv: int
    num_reward: int
    num_workers: int
    reward_decay: float
    rnn_layers: int
    # Key: step size, value: number of steps.
    seq_lens: Dict[int, int]
    state_size: int
    # Number of steps to sample from each trajectory per epoch.
    steps_per_epoch: int


class Config(NamedTuple):
    """Training config."""
    cuda: bool
    environment: str
    seed: Optional[int]
    skip_failed: bool

    state: StateConfig
    trans: TransConfig


def _to_dict(conf: NamedTuple):
    """Convert nested named tuple to dict."""
    d = conf._asdict()
    for k in conf._fields:
        el = getattr(conf, k)
        if hasattr(el, '_asdict'):
            d[k] = _to_dict(el)
    return d


def load_config(p: Path) -> Config:
    with open(p) as f:
        d = yaml.load(f, Loader=yaml.SafeLoader)
    state = StateConfig(**d['state'])
    del d['state']
    trans = TransConfig(**d['trans'])
    del d['trans']
    return Config(state=state, trans=trans, **d)


def save_config(conf: Config, dir_path: Path):
    """Save a config to a directory."""
    os.makedirs(dir_path, exist_ok=True)
    with open(dir_path / 'agent_train_conf.yaml', 'w') as f:
        yaml.dump(_to_dict(conf), f, Dumper=yaml.SafeDumper)


def override_config(conf: NamedTuple, **kwargs):
    """Allow overwriting config attributes selectively."""
    if len(kwargs) == 0:
        return conf
    else:
        for k in kwargs:
            el = getattr(conf, k)
            if hasattr(el, '_asdict'):
                kwargs[k] = override_config(el, **kwargs[k])

        d = conf._asdict()
        ConfType = type(conf)
        return ConfType(**{**d, **kwargs})

