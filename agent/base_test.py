from pathlib import Path
import logging
from unittest import TestCase

from agent.config import Config, load_config, override_config


logging.basicConfig(level=logging.DEBUG)


class BaseTest(TestCase):
    repo_root = Path(__file__).parents[1]
    data_dir = repo_root / 'data'
    _train_conf = None

    @classmethod
    def get_conf(cls, **kwargs) -> Config:
        """Allow overwriting train config arguments"""
        if cls._train_conf is None:
            cls._train_conf = load_config(cls.repo_root / 'train_conf.yaml')
        kwargs['cuda'] = False
        return override_config(cls._train_conf, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
