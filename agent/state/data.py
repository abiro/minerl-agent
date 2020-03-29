import logging
from collections import defaultdict
import gc
from itertools import chain
from pathlib import Path
from typing import Iterable, NamedTuple, List, Dict, Tuple, Callable

import cv2
import numpy as np
import torch as T
import torch.utils.data as tdata

from agent.config import Config, StateConfig


class Stats(NamedTuple):
    mean: np.ndarray
    std: np.ndarray


class Sample(NamedTuple):
    # Sequences of observations, actions and frames. The key is the offset
    # relative to the current time step backwards in time.
    # Observations contains:
    # - One-hot encoding of inventory +
    # - One-hot encoding of the currently equipped item
    inv: Dict[int, np.ndarray]  # (num_inv + num_equipable,)
    actions: Dict[int, np.ndarray]  # (num_act,)
    frames: Dict[int, np.ndarray]  # (n_channels, width, height)

    # Trajectory name
    trajectory: str


class Batch:
    def __init__(self, inv: Dict[int, T.Tensor],
                 actions: Dict[int, T.Tensor],
                 frames: Dict[int, T.Tensor],
                 trajectories: List[str]):
        # Sequences of inventory, actions, frames and rewards. The key is
        # the offset relative to the current time step.
        # (batch_size, num_inv + num_equipable,)
        self.inv = inv
        # (batch_size, num_act,)
        self.actions = actions
        # (batch_size, n_channels, width, height)
        self.frames = frames
        # List of trajectory names for each member of the batch
        # (for debugging purposes)
        self.trajectories = list(trajectories)

    def pin_memory(self) -> 'Batch':
        for key in self.inv:
            self.inv[key].pin_memory()
            self.actions[key].pin_memory()
            self.frames[key].pin_memory()

        return self

    def to(self, device: T.device, non_blocking: bool = False) -> 'Batch':
        nb = non_blocking
        for key in self.inv:
            self.inv[key] = self.inv[key].to(device, non_blocking=nb)
            self.actions[key] = self.actions[key].to(device,
                                                     non_blocking=nb)
            self.frames[key] = self.frames[key].to(device, non_blocking=nb)

        return self


class Dataset(tdata.IterableDataset):
    _meta_f = Path('metadata.json')
    _recording_f = Path('recording.mp4')
    _obs_f = Path('rendered.npz')
    _equip_type = (
        'none',
        'air',
        'wooden_axe',
        'wooden_pickaxe',
        'stone_axe',
        'stone_pickaxe',
        'iron_axe',
        'iron_pickaxe',
        'other')
    _inv_cols = (
        'coal',
        'cobblestone',
        'crafting_table',
        'dirt',
        'furnace',
        'iron_axe',
        'iron_ingot',
        'iron_ore',
        'iron_pickaxe',
        'log',
        'planks',
        'stick',
        'stone',
        'stone_axe',
        'stone_pickaxe',
        'torch',
        'wooden_axe',
        'wooden_pickaxe',
        'diamond',
    )
    _rewards = {
        'diamond': 1024
    }

    # TODO mixed precision
    dtype = np.float32
    ttype = T.float32

    @staticmethod
    def get_indices(seq_lens: Dict[int, int]) -> List[int]:
        indices = set()
        for step_size, num_steps in seq_lens.items():
            if num_steps < 0:
                raise ValueError('num_steps can not be negative: {}'.format(
                    num_steps))
            elif num_steps == 0:
                continue
            for j in range(num_steps):
                indices.add(step_size * j)
        return sorted(indices)

    @staticmethod
    def _get_step_weights(rewards: np.ndarray, decay: float) -> np.ndarray:
        if rewards.ndim != 1:
            raise ValueError('Expected 1 dim array, instead got: {}'.format(
                rewards.ndim))
        weights = np.ones_like(rewards, dtype=np.float)
        rew_mask = rewards > 0
        # Rewards are powers of 2 starting with 1. We want reward 1 to have
        # weight 2 etc.
        weights[rew_mask] = np.log2(rewards[rew_mask]) + 2

        # Apply reward decay
        for i in reversed(range(weights.size - 1)):
            if not rew_mask[i]:
                weights[i] = max(1, weights[i+1] * decay)

        return weights

    @classmethod
    def _load_observations(cls, traj_dir: Path) -> Dict[str, np.ndarray]:
        with np.load(traj_dir / cls._obs_f) as f:
            d = dict(f.items())
            # First step doesn't have reward, but we want it to be the same
            # length as the inventory observations
            d['reward'] = np.insert(d['reward'], 0, [0])
            return d

    @classmethod
    def _to_onehot(cls, items: np.ndarray, num_discrete: int):
        if num_discrete < 1:
            raise ValueError('num_discrete must be greater than 0')
        if items.ndim != 1:
            s = items.shape
            raise ValueError('Expected one dim array, got: {}'.format(s))
        one_hot = np.zeros((len(items), num_discrete), dtype=cls.dtype)
        for i, val in enumerate(items.astype(np.int)):
            one_hot[i, val] = 1
        return one_hot

    @classmethod
    def _parse_actions(cls, observations: Dict[str, np.ndarray]):
        actions = []
        steps = None
        for key in sorted(observations.keys()):
            if key.startswith('action'):
                a = observations[key]
                if steps is not None and a.shape[0] != steps:
                    raise ValueError('num action mismatch')
                steps = a.shape[0]
                # TODO discretize camera
                # camera is Box(2,)
                if len(a.shape) > 2:
                    s = 'Unexpected shape key: {} shape: {}'.format(
                        key, a.shape)
                    raise ValueError(s)

                if key == 'action_craft':
                    a = cls._to_onehot(a, 5)
                elif key == 'action_equip':
                    # equip action doesn't have "other"
                    n_discrete = len(cls._equip_type) - 1
                    a = cls._to_onehot(a, n_discrete)
                elif key == 'action_nearbyCraft':
                    a = cls._to_onehot(a, 8)
                elif key == 'action_nearbySmelt':
                    a = cls._to_onehot(a, 3)
                elif key == 'action_place':
                    a = cls._to_onehot(a, 7)

                if len(a.shape) == 2:
                    for i in range(a.shape[1]):
                        actions.append(a[:, i])
                else:
                    actions.append(a)

        num_act = len(actions)
        # Last step is missing actions
        a = np.zeros((steps + 1, num_act), dtype=cls.dtype)
        a[:-1] = np.stack(actions, axis=-1)
        pad = np.zeros((num_act,), dtype=cls.dtype)
        a[-1] = pad
        return pad, a

    @classmethod
    def _parse_reward(cls, observations: Dict[str, np.ndarray],
                      num_reward: int):
        r = observations['reward']
        r_mask = r > 0
        scaled_r = np.zeros_like(r, dtype=np.float)
        scaled_r[r_mask] = np.log2(r[r_mask])
        rewards = cls._to_onehot(scaled_r, num_reward)
        pad = np.zeros((num_reward,), dtype=cls.dtype)
        return pad, rewards

    @classmethod
    def _load_frames(cls, traj_dir: Path):
        p = traj_dir / cls._recording_f
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise FileNotFoundError(p)
        try:
            frame_shape = None
            frames = []
            while True:
                ret, im = cap.read()
                if not ret:
                    break
                if frame_shape is not None and im.shape != frame_shape:
                    raise ValueError('Frame shape mismatch {}'.format(
                        traj_dir))
                frame_shape = im.shape
                # Rescale frames ]-1, 1]
                im = (im - 127.) / 128.
                frames.append(im)

            # stack makes copies
            a = np.stack(frames).astype(cls.dtype)
            # Pytorch conv layers need channels first
            a = np.transpose(a, (0, 3, 1, 2))
            pad = np.zeros(shape=a.shape[1:], dtype=cls.dtype)

            return pad, a
        finally:
            cap.release()

    @classmethod
    def collate_fn(cls, samples: Iterable['Dataset.Sample']) \
            -> 'Dataset.Batch':
        """Turn a sequence of samples into a batch"""
        inv = defaultdict(list)
        actions = defaultdict(list)
        frames = defaultdict(list)
        trajectories = []
        for s in samples:
            trajectories.append(s.trajectory)
            for ri in s.inv:
                inv[ri].append(s.inv[ri])
                frames[ri].append(s.frames[ri])
                actions[ri].append(s.actions[ri])

        inv_out = {}
        actions_out = {}
        frames_out = {}
        for ri in inv:
            inv_out[ri] = T.tensor(np.stack(inv[ri], axis=0),
                                   dtype=cls.ttype)
            frames_out[ri] = T.tensor(np.stack(frames[ri], axis=0),
                                      dtype=cls.ttype)
            actions_out[ri] = T.tensor(np.stack(actions[ri], axis=0),
                                       dtype=cls.ttype)

        return Batch(inv=inv_out,
                     actions=actions_out,
                     frames=frames_out,
                     trajectories=trajectories)

    def __init__(self, datadir: Path, conf: Config):
        super().__init__()
        self._dir = datadir / conf.environment
        if not self._dir.is_dir():
            raise FileNotFoundError(self._dir)

        self._env = conf.environment
        self._conf_seed = conf.seed
        self._num_reward = conf.state.num_reward
        self._reward_decay = conf.state.reward_decay
        self._relative_indices = self.get_indices(conf.state.seq_lens)
        self._steps_per_epoch = conf.state.steps_per_epoch
        self._skip_failed = conf.skip_failed

        self._inv_stats_h = None
        self._rand_h = None
        self._traj_h = None
        self._epoch = 1

    def __iter__(self) -> Iterable[Sample]:
        worker_info = tdata.get_worker_info()
        if worker_info:
            step_size = worker_info.num_workers
            worker_id = worker_info.id
        else:
            step_size = 1
            worker_id = 0

        # Split trajectories across workers
        trajectories = self._shuffle_trajectories()
        for i in range(worker_id, len(trajectories), step_size):
            traj_dir = trajectories[i]
            observations = self._load_observations(traj_dir)
            if self._skip_failed and \
                    not self._check_success(observations['reward']):
                continue

            sampled_steps = self._get_steps(observations['reward'])

            a_pad, actions = self._parse_actions(observations)
            n_steps = actions.shape[0]
            i_pad, inv = self._standardize_inv(self._parse_inv(observations))
            assert n_steps == inv.shape[0], traj_dir
            f_pad, frames = self._load_frames(traj_dir)
            if frames.shape[0] != n_steps:
                m = 'Num frames: {} num steps: {} trajectory: {}'
                logging.debug(m.format(frames.shape[0], n_steps, traj_dir))
            # Sometimes an episode starts after a number of frames have been
            # recorded.
            frames = frames[-n_steps:]

            msg = ('Worker {} loaded trajectory: {}. Total steps: {} '
                   'sampled steps: {}')
            logging.debug(msg.format(worker_id, traj_dir.name, n_steps,
                                     sampled_steps.size))

            for j in sampled_steps:
                s_inv = {}
                s_actions = {}
                s_frames = {}
                for ri in self._relative_indices:
                    if ri > j:
                        s_inv[ri] = i_pad
                        s_actions[ri] = a_pad
                        s_frames[ri] = f_pad
                    else:
                        s_inv[ri] = inv[j-ri]
                        s_actions[ri] = actions[j-ri]
                        s_frames[ri] = frames[j-ri]
                yield Sample(inv=s_inv,
                             actions=s_actions,
                             frames=s_frames,
                             trajectory=traj_dir.name)

            gc.collect()
        self._epoch += 1

    @property
    def _inv_stats(self) -> Stats:
        if self._inv_stats_h is None:
            invs = np.concatenate([inv for inv in self.iter_inv()], axis=0)
            mean = np.mean(invs, axis=0)
            std = np.std(invs, axis=0)
            self._inv_stats_h = Stats(mean=mean, std=std)
        return self._inv_stats_h

    @property
    def _rand(self) -> np.random.RandomState:
        # Random state should be initialized in worker process
        if self._rand_h is None:
            worker_info = tdata.get_worker_info()
            if worker_info:
                # Numpy requires int32 seeds, but Pytorch sometimes sets int64
                seed = worker_info.seed % 1000003
            else:
                seed = self._conf_seed
            self._rand_h = np.random.RandomState(seed)
        return self._rand_h

    @property
    def _trajectories(self) -> List[Path]:
        if self._traj_h is None:
            self._traj_h = [p for p in self._dir.iterdir() if p.is_dir()]
        return self._traj_h

    def _check_success(self, rewards: np.ndarray) -> bool:
        if 'Diamond' not in self._env:
            raise NotImplementedError('Not implemented for {}'.format(
                self._env))
        # The success property of the metadata is often wrong
        return np.any(rewards == self._rewards['diamond'])

    def _get_steps(self, rewards: np.ndarray) -> np.ndarray:
        """Sample steps based on reward density with replacement.

        Args:
            rewards: Rewards for each time step in the trajectory.
                Shape: (total_steps,)
        """
        weights = self._get_step_weights(rewards, self._reward_decay)
        p = weights / np.sum(weights)
        steps = self._rand.choice(np.arange(rewards.shape[0]),
                                  self._steps_per_epoch,
                                  replace=True,
                                  p=p)

        return steps

    def _parse_inv(self, observations: Dict[str, np.ndarray]):
        inventory = observations['observation_inventory']
        equipped = observations['observation_type']
        damage = observations['observation_damage'].reshape((-1, 1))
        max_damage = observations['observation_maxDamage'].reshape((-1, 1))
        eq_one_hot = self._to_onehot(equipped, len(self._equip_type))

        obs = np.concatenate([inventory, damage, max_damage, eq_one_hot],
                             axis=1)
        obs = obs.astype(self.dtype)
        return obs

    def _shuffle_trajectories(self) -> List[Path]:
        trajectories = list(self._trajectories)
        # Trajectory order must be same across all workers
        rand = np.random.RandomState(1000 + self._epoch)
        rand.shuffle(trajectories)
        return trajectories

    def _standardize_inv(self, obs):
        stats = self._inv_stats
        std_mask = stats.std != 0
        std = np.ones_like(stats.std)
        std[std_mask] = stats.std[std_mask]
        obs = (obs - stats.mean) / std

        # Pad values should correspond to 0 before standardization
        pad = (-stats.mean) / std
        pad = pad.astype(self.dtype)
        return pad, obs

    def iter_inv(self) -> Iterable[np.ndarray]:
        """Iterate over targets for all the trajectories."""
        for traj_dir in self._trajectories:
            observations = self._load_observations(traj_dir)
            rewards = observations['reward']
            if self._skip_failed and self._check_success(rewards):
                yield self._parse_inv(observations)

    def warm_up(self) -> 'Dataset':
        """Initialize members that require reading from disk.

        These members are lazy initialized as initializing them in init would
        slow down testing too much, but it doesn't make sense to repeat these
        tasks in every worker, hence this method which is intended to be
        called before the dataset is passed to a `torch.utils.data.DataLoader`
        constructor.
        """
        self._trajectories
        self._inv_stats
        return self


class DummyDataset(tdata.IterableDataset):

    def __init__(self, conf: Config, max_steps: int):
        super().__init__()

        self._max_steps = max_steps
        self._relative_indices = Dataset.get_indices(conf.state.seq_lens)
        self._num_inv = conf.state.num_inv
        self._num_act = conf.state.num_act

    def __iter__(self) -> Iterable[Sample]:
        for i in range(self._max_steps):
            inv = {}
            frames = {}
            actions = {}
            for ri in self._relative_indices:
                inv[ri] = np.zeros((self._num_inv,), dtype=np.float)
                actions[ri] = np.zeros((self._num_act,), dtype=np.float)
                frames[ri] = np.zeros((3, 64, 64), dtype=np.float)

            last_i = self._relative_indices[-1]
            a = i % 3
            if a == 0:
                # Test if info can flow from frame to inv
                frames[last_i][a] = 1.
                inv[0][a] = 1.
            elif a == 1:
                # Test if info can flow from action to frame
                actions[last_i][a] = 1.
                frames[0][a] = 1.
            else:
                # Test if info can flow from action to inv
                actions[last_i][a] = 1.
                inv[0][a] = 1.

            yield Sample(inv=inv, actions=actions, frames=frames,
                         trajectory='dummy')

    def warm_up(self) -> 'DummyDataset':
        return self


class BatchLoader:
    """Wrapper over PyTorch DataLoader.

    Ensures that each batch contains samples from all workers.
    """
    @staticmethod
    def _identity(items):
        return items

    def __init__(self, dataset: Dataset, sconf: StateConfig,
                 collate_fn: Callable, pin_memory: bool = True):
        if sconf.steps_per_epoch % sconf.batch_size != 0:
            raise ValueError('Steps per epoch must be divisible by batch '
                             'size.')
        self._n_workers = sconf.num_workers
        if self._n_workers > 0:
            if sconf.batch_size % self._n_workers != 0:
                raise ValueError('Batch size must be divisible by num workers')
            per_worker = sconf.batch_size // sconf.num_workers
        else:
            per_worker = sconf.batch_size
        self._collate_fn = collate_fn
        self._pin_memory = pin_memory
        self._loader = tdata.DataLoader(dataset,
                                        batch_size=per_worker,
                                        num_workers=sconf.num_workers,
                                        # collating and pinning is done in iter
                                        collate_fn=self._identity,
                                        pin_memory=False)

    def __iter__(self) -> Iterable[Batch]:
        if self._n_workers == 0:
            for samples in self._loader:
                batch = self._collate_fn(samples)
                if self._pin_memory:
                    yield batch.pin_memory()
                else:
                    yield batch
        else:
            yield from self._iter_multi()

    def _iter_multi(self) -> Iterable[Batch]:
        loader_it = iter(self._loader)
        while True:
            samples = []
            for _ in range(self._n_workers):
                try:
                    samples.append(next(loader_it))
                except StopIteration:
                    pass
            if len(samples) == 0:
                break
            else:
                batch = self._collate_fn(chain(*samples))
                if self._pin_memory:
                    yield batch.pin_memory()
                else:
                    yield batch

