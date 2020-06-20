from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys, select
import os.path as osp
import random
import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_currenttime_prefix():
    """to get a prefix of current time
    
    Returns:
        [str] -- current time encoded into string
    """

    from time import localtime, strftime

    return strftime("%d-%b-%Y_%H:%M:%S", localtime())

def get_timed_input(string, timeout=5):
    print("{} (waiting {} seconds) : ".format(string, timeout))
    i, o, e = select.select( [sys.stdin], [], [], 10 )
    in_string = ''
    if (i):
        in_string = sys.stdin.readline().strip()
    return in_string
