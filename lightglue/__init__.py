import importlib.resources as resources

from .aliked import ALIKED  # noqa
from .disk import DISK  # noqa
from .dog_hardnet import DoGHardNet  # noqa
from .lightglue import LightGlue  # noqa
from .sift import SIFT  # noqa
from .utils import match_pair  # noqa

PRETRAINED_MODEL_WEIGHTS_PATH = resources.files('lightglue.weights')