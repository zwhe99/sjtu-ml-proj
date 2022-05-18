import os
import random
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import torch


# define color

# red-transparent-blue
colors = []
for l in np.linspace(1, 0, 1000):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 1000):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

# jet-transparent
colors = plt.get_cmap('jet')(range(256))
colors[:,-1] = np.linspace(0.2, 1, 256)
jet_transparent = LinearSegmentedColormap.from_list(name='jet_transparent',colors=colors)

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
