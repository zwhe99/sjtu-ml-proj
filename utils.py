import os
import random

import numpy as np
import torch


def pad_sequence(sequences, left_pad, padding_value):
    max_len = max([len (sq) for sq in sequences])

    if not left_pad:
        return [sq + [padding_value] * (max_len - len(sq)) for sq in sequences]
    else:
        return [[padding_value] * (max_len - len(se)) + se for se in sequences]

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
