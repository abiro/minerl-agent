import os
from collections import defaultdict
import logging
from operator import itemgetter
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Iterable, Iterator
import time

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


from agent.config import Config, StateConfig
from agent.state.data import Batch, BatchLoader, Dataset


class FrameEncoder(nn.Module):

    def __init__(self, conf: StateConfig):
        super().__init__()
        # TODO how does batch norm behave when repeatedly applied before back
        # prop
        self._main = nn.Sequential(
            # 32x32 out for 64x64 input
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                      padding=1),  # noqa 501
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 16, 1),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),

            # 16x16 out
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 64, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(64, 32, 1),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),

            # 8x8 out
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 128, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(128, 64, 1),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),

            # 4x4 out
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 256, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),

            # 2x2 out
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(256, 128, 1),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),

            # 1x1 out
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(256, conf.cnn_out, 1),
            nn.LeakyReLU(True),
            nn.MaxPool2d(2),
        )

    def forward(self, frame: T.Tensor) -> T.Tensor:
        return self._main(frame).flatten(start_dim=1)


class InvEncoder(nn.Module):
    def __init__(self, conf: StateConfig):
        super().__init__()

        self._main = nn.Sequential(
            # Inv observations are standardized in data loader
            nn.Linear(conf.num_inv, 2 * conf.num_inv),
            nn.LeakyReLU(inplace=True),
            nn.Linear(2 * conf.num_inv, conf.inv_out),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, features: T.Tensor) -> T.Tensor:
        return self._main(features)


class StateEncoder(nn.Module):
    @staticmethod
    def _get_input_indices(step_size: int, num_steps: int) -> List[int]:
        """Get the input indices for one of the step sizes.

        Args:
            step_size: Exponent for the step size.
            num_steps: The number of steps.

        Returns:
            The input indices in reversed where t_0 is always the last. Eg. for
                i=1 and num_steps=2 the output is [2, 0].

        Raises:
            ValueError for i < 0 or num_steps < 0
        """
        if step_size < 0 or num_steps < 0:
            raise ValueError(('Input must be non-negative. step_size: {} '
                              'num_steps: {}').format(step_size, num_steps))
        stop = step_size * num_steps
        return list(reversed(range(0, stop, step_size)))

    def __init__(self, conf: StateConfig):
        super().__init__()

        if len(conf.seq_lens) > 1:
            raise NotImplementedError('Only implemented for one step size')

        self._seq_lens = dict(conf.seq_lens)
        self._out_size = conf.state_size
        self._num_act = conf.num_act

        self._frame_encoder = FrameEncoder(conf)
        self._inv_encoder = InvEncoder(conf)

        self._input_indices = {}
        flat_indices = set()
        for step_size, num_steps in self._seq_lens.items():
            ii = self._get_input_indices(step_size, num_steps)
            self._input_indices[step_size] = ii
            flat_indices.update(self._input_indices[step_size])
        # Index 0 is the current time step which we want last
        self._flat_indices = sorted(flat_indices, reverse=True)

        rnn_in = conf.cnn_out + conf.inv_out + conf.num_act
        self._bn = nn.BatchNorm1d(rnn_in)
        self._rnn = nn.GRU(input_size=rnn_in,
                           hidden_size=conf.state_size,
                           num_layers=conf.rnn_layers)
        self._skip_merge = nn.ModuleDict()

    @property
    def out_size(self):
        return self._out_size

    def forward(self, batch: Batch) -> T.Tensor:
        """Compute a forward pass.

        Args:
            batch: Batch

        Returns:
            The state encoding for each time step.

        """

        inputs = []
        for i in self._flat_indices:
            f_out = self._frame_encoder(batch.frames[i])
            i_out = self._inv_encoder(batch.inv[i])
            joined = T.cat([f_out, i_out, batch.actions[i]], dim=-1)
            inputs.append(self._bn(joined))

        # TODO skip connection from inputs to state
        inputs_t = T.stack(inputs, dim=0)
        state, _ = self._rnn(inputs_t)

        # State embeddings for t+1 from each time step t
        return state


class FrameGenerator(nn.Module):
    """DCGAN Generator for frame generation from state.

    Based on: https://github.com/pytorch/examples/blob/master/dcgan/main.py
    """
    def __init__(self, conf: StateConfig):
        super().__init__()
        ngf = conf.frame_gen_filters
        self.main = nn.Sequential(
            # (ngf*8) x 4 x 4 out
            nn.BatchNorm2d(conf.state_size),
            nn.ConvTranspose2d(in_channels=conf.state_size,
                               out_channels=ngf * 8,
                               kernel_size=4,
                               stride=1,
                               padding=0),
            nn.LeakyReLU(inplace=True),

            # (ngf*4) x 8 x 8 out
            nn.BatchNorm2d(ngf * 8),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(True),

            # (ngf*2) x 16 x 16 out
            nn.BatchNorm2d(ngf * 4),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.LeakyReLU(True),

            # ngf x 32 x 32 out
            nn.BatchNorm2d(ngf * 2),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.LeakyReLU(True),

            # 3 x 64 x 64 out
            nn.BatchNorm2d(ngf),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1),
        )

    def forward(self, state: T.Tensor) -> T.Tensor:
        """Computes a forward pass

        Args:
            state: (batch_size, state_size)

        Returns:
            (batch_size, 3, 64, 64)

        """
        s = state.view((state.shape[0], state.shape[1], 1, 1))
        return self.main(s)


class InvGenerator(nn.Module):
    """Generate inventory features from state."""
    def __init__(self, conf: StateConfig):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm1d(conf.state_size),
            nn.Linear(conf.state_size, 2 * conf.num_inv),
            nn.LeakyReLU(inplace=True),

            nn.BatchNorm1d(2 * conf.num_inv),
            nn.Linear(2 * conf.num_inv, 2 * conf.num_inv),
            nn.LeakyReLU(inplace=True),

            nn.BatchNorm1d(2 * conf.num_inv),
            nn.Linear(2 * conf.num_inv, conf.num_inv),
        )

    def forward(self, state: T.Tensor) -> T.Tensor:
        """Computes a forward pass

        Args:
            state: (batch_size, state_size)

        Returns:
            (batch_size, num_inv)

        """
        return self.main(state)


class Generator(nn.Module):
    """GAN Generator"""
    def __init__(self, conf: StateConfig):
        super().__init__()
        self._frame_gen = FrameGenerator(conf)
        self._inv_gen = InvGenerator(conf)

    def forward(self, state: T.Tensor) -> Tuple[T.Tensor, T.Tensor]:
        return self._frame_gen(state), self._inv_gen(state)


class Critic(nn.Module):
    """DCGAN Critic."""
    def __init__(self, conf: StateConfig, encoder: StateEncoder):
        super().__init__()
        self.encoder = encoder
        self.main = nn.Sequential(
            nn.BatchNorm1d(conf.state_size),
            nn.Linear(conf.state_size, 2 * conf.state_size),
            nn.LeakyReLU(inplace=True),

            nn.Linear(2 * conf.state_size, 1),
        )

    def forward(self, batch: Batch) -> Tuple[T.Tensor, T.Tensor]:
        state = self.encoder(batch)
        # state[-1] is last time step which is step 0 in the batch
        # dicts
        pred = self.main(state[-1]).mean()
        return state, pred


# TODO use reward from dense environment as signal. Problem: reward will be
#  sparse in evaluation environment, so it can't be input to encoder.
class Training:
    @staticmethod
    def _ensure_dir(dir_path: Path) -> Path:
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    @staticmethod
    def _save_model(ckpt_dir: Path, epoch: int, model: nn.Module,
                    optimizer: Optional[optim.Optimizer] = None):
        fname = 'epoch_{:03d}.tar'.format(epoch)
        p = ckpt_dir / fname
        d = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }
        if optimizer is not None:
            d['optimizer_state_dict'] = optimizer.state_dict(),
        T.save(d, p)
        logging.info('Saved checkpoint to: {}'.format(p))

    def __init__(self, dataset: Dataset, conf: Config, device: T.device,
                 ckpt_dir: Path, write_freq: int = 100,
                 writer: Optional[SummaryWriter] = None):
        self.sconf = conf.state
        self.encoder_ckpt = self._ensure_dir(ckpt_dir / 'encoder')
        self.generator_ckpt = self._ensure_dir(ckpt_dir / 'generator')
        self.critic_ckpt = self._ensure_dir(ckpt_dir / 'critic')

        self.device = device
        self.encoder = StateEncoder(self.sconf).train().to(device)
        self.generator = Generator(self.sconf).train().to(device)
        self.critic = Critic(self.sconf, self.encoder).train().to(device)
        self.gen_opt = optim.RMSprop(self.generator.parameters(),
                                     lr=self.sconf.lr)
        self.crit_opt = optim.RMSprop(self.critic.parameters(),
                                      lr=self.sconf.lr)

        self.loader = BatchLoader(dataset.warm_up(),
                                  conf.state,
                                  Dataset.collate_fn,
                                  pin_memory=True)
        self.writer = writer

        self._one = T.tensor(1.).to(device)
        self._mone = T.tensor(-1.).to(device)
        self._write_freq = write_freq
        self._global_step = 1
        self._epoch = 1
        self._epoch_start = 0
        self._epoch_end = 0
        self._train_end = False
        self._metrics = defaultdict(list)

    def _save_all(self):
        self._save_model(self.encoder_ckpt, self._epoch, self.encoder)
        self._save_model(self.generator_ckpt, self._epoch, self.generator,
                         self.gen_opt)
        self._save_model(self.critic_ckpt, self._epoch, self.critic,
                         self.crit_opt)

    def _flow_batches(self):
        self._on_train_start()
        self._global_step = 0
        for i in range(1, self.sconf.epochs + 1):
            self._epoch = i
            self._on_epoch_start()
            for batch in self.loader:
                # We want the first train step to be 1, but it shouldn't be
                # incremented after the last batch of the last epoch.
                self._global_step += 1
                yield batch.to(self.device, non_blocking=True)
                self._on_step_end()
            self._on_epoch_end()
        self._on_train_end()

    def _flush_metrics(self):
        tmpl = '| {}: {:.5f}'
        messages = []

        mean_metrics = {key: np.mean(vals) for key, vals in
                        self._metrics.items() if len(vals) > 0}
        for key, val in sorted(mean_metrics.items(), key=itemgetter(0)):
            if self.writer is not None:
                self.writer.add_scalar(key, val, self._global_step)
            else:
                messages.append(tmpl.format(key, val))

        if len(messages) > 0:
            msg = '|step: {:05d}'.format(self._global_step)
            msg = ' '.join([msg] + messages)
            logging.info(msg)

        self._metrics = defaultdict(list)

    def _on_epoch_start(self):
        self._epoch_start = time.perf_counter()

    def _on_epoch_end(self):
        epoch_time = round((time.perf_counter() - self._epoch_start) / 60, 2)
        logging.info('Epoch {} complete in {} m'.format(self._epoch,
                                                        epoch_time))
        if self._epoch % self.sconf.checkpoint_freq == 0:
            self._save_all()

    def _on_step_end(self):
        if self._global_step % self._write_freq == 0:
            self._flush_metrics()

    def _on_train_start(self):
        self._train_end = False
        self._metrics = defaultdict(list)

    def _on_train_end(self):
        self._train_end = True
        if self._global_step % self._write_freq != 0:
            self._flush_metrics()
        if self._epoch % self.sconf.checkpoint_freq != 0:
            self._save_all()

    def _get_grad_penalty(self):
        # Based on https://github.com/caogang/wgan-gp
        raise NotImplementedError

    def _train_critic(self, batches: Iterator):
        if self._epoch == 1:
            n_iters = self.sconf.critic_turbo_iters
        else:
            n_iters = self.sconf.critic_iters

        for i in range(n_iters):
            try:
                batch = next(batches)
            except StopIteration:
                break

            self.crit_opt.zero_grad()
            # TODO gradient penalty instead of clamping
            for p in self.critic.parameters():
                p.data.clamp_(-self.sconf.critic_clamp,
                              self.sconf.critic_clamp)

            real_state, real_pred = self.critic(batch)
            real_pred.backward(self._mone)

            with T.no_grad():
                # real_state[-2] is the state prediction for t_0
                fake_frames, fake_invs = self.generator(real_state[-2])
            batch.frames[0] = fake_frames
            batch.inv[0] = fake_invs
            _, fake_pred = self.critic(batch)
            fake_pred.backward(self._one)

            self.crit_opt.step()

            # Wasserstein distance between real and fake distribution.
            # The larger its magnitude, the better the critic.
            # The optimizer minimizes, so the distance is negative, but it
            # makes more sense to display it as a positive number, hence the
            # abs.
            wd = abs(fake_pred.item() - real_pred.item())
            self._metrics['wd'].append(wd)

    def _train_generator(self, batches: Iterator) -> \
            Optional[Tuple[float, float]]:
        try:
            batch = next(batches)
        except StopIteration:
            return None

        self.gen_opt.zero_grad()

        with T.no_grad():
            real_state = self.encoder(batch)
        # real_state[-2] is the state prediction for t_0
        fake_frames, fake_invs = self.generator(real_state[-2])
        real_frames, real_invs = batch.frames[0], batch.inv[0]
        batch.frames[0] = fake_frames
        batch.inv[0] = fake_invs
        _, fake_pred = self.critic(batch)
        fake_pred.backward(self._mone)

        self.gen_opt.step()

        # The lower the error, the better the generator
        self._metrics['err_g'].append(abs(fake_pred.item()))

        with T.no_grad():
            l1_frame = F.l1_loss(fake_frames, real_frames).cpu().item()
            l1_inv = F.l1_loss(fake_invs, real_invs).cpu().item()
        self._metrics['l1_frame'].append(l1_frame)
        self._metrics['l1_inv'].append(l1_inv)
        last_l1_frame = self._metrics['l1_frame'][-1]
        last_l1_inv = self._metrics['l1_inv'][-1]

        return last_l1_frame, last_l1_inv

    def train(self) -> Tuple[float, float]:
        # Critic and generator take different number of steps, so we need a
        # mechanism to yield steps_per_epoch * epoch batches, while
        # keeping track of the current global step and epoch.
        batch_flow = self._flow_batches()

        l1_scores = (-1., -1.)
        while not self._train_end:
            self._train_critic(batch_flow)
            res = self._train_generator(batch_flow)
            if res:
                l1_scores = res
        return l1_scores

