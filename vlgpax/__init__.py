from jax.config import config
config.update('jax_enable_x64', True)

from .model import Session
from .vi import *
