from enum import Enum


class Loss(str, Enum):
    L1 = "l1"
    SCI = "sci"
    NLL = "nll"


class Transform(str, Enum):
    DEVELOPMENT = "development"
    FLIP = "flip"
    LLFLOW = "LLFlow"
    FLIP_NO_RESIZE = "flip_no_resize"


class Optimizer(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class Scheduler(str, Enum):
    CONSTANT = "constant"
    COSINE = "cosine"
    ONE_CYCLE = "one_cycle"


class PairSelectionMethod(str, Enum):
    """Pair selection method for SICE dataset.

    RANDOM_NEXT: Select a random image from a sequence and its successor.
    RANDOM_TARGET: Select a random image from a sequence and a ground truth image.
    HALFEXP_TARGET: Select a -1ev image from a sequence and a ground truth image
    """

    RANDOM_NEXT = "random_next"
    RANDOM_TARGET = "random_target"
    HALFEXP_TARGET = "halfexp_target"


class SupplementaryDataset(str, Enum):
    DICM = "dicm"
    FUSION = "fusion"
    LIME = "lime"
    LOW = "low"
    MEF = "mef"
    NPE = "npe"
    VV = "vv"
