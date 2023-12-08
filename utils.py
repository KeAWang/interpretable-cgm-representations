def seed_everything(seed: int):
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def to_numpy(x):
    return x.detach().cpu().numpy()

def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


blue, red, turquoise, purple, magenta, orange, gray  = [[0.372549, 0.596078, 1], 
                                                  [1.0, .3882, .2784], 
                                                  [0.20784314, 0.67843137, 0.6], 
                                                  [0.59607843, 0.25882353, 0.89019608],
                                                  [0.803922, 0.0627451, 0.462745], 
                                                  [0.917647, 0.682353, 0.105882],
                                                  [0.7, 0.7, 0.7]
                                                  ]
PALETTE_DICT = {
    "blue": blue,
    "red": red,
    "turquoise": turquoise,
    "purple": purple,
    "magenta": magenta,
    "orange": orange,
    "gray": gray,
}
palette = list(PALETTE_DICT.values())