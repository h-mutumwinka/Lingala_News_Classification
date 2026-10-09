import os
import random
import sys

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Only seed torch if the notebook has already imported it
    if "torch" in sys.modules:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
