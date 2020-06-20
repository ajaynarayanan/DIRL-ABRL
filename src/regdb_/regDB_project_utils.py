from __future__ import absolute_import
from __future__ import division
from collections import OrderedDict
import os, sys, argparse, time
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os.path as osp, pickle
from dataset_regdb import Dataset
from my_utils import *


def determine_save_path(opt):
    # if there is no pretrained model given, init new save folder path
    if opt.pretrained_model == "":
        time_prefix = get_currenttime_prefix()
        save_prefix = get_timed_input(
            "enter the folder name to save (default:{}) ".format(opt.save_prefix),
            timeout=15,
        )
        if save_prefix != "":
            opt.save_prefix = save_prefix

        opt.save_root = osp.join(
            osp.abspath(opt.save_root), time_prefix + "_" + opt.save_prefix
        )
    else:
        opt.save_root = osp.split(osp.abspath(opt.pretrained_model))[0]

    print("all models and logs will be stored in {}".format(opt.save_root))
    if not osp.exists(opt.save_root):
        os.mkdir(opt.save_root)


def init_logger_(opt):
    log_file_path = osp.join(opt.save_root, "train.log")
    return Logger(fpath = log_file_path)


def get_dataset(opt):

    # specify the rgb, IR transform
    train_dataset = Dataset(
        opt.dataroot,
        "train",
        opt.trial,
        attr_mode = True,
    )
    return train_dataset


def OptionsParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs of training"
    )
    parser.add_argument(
        "--train-batch", type=int, default=64, help="size of the batches"
    )
    parser.add_argument(
        "--test-batch", type=int, default=64, help="size of the batches"
    )
    
    parser.add_argument(
        "--dataroot",
        type=str,
        default= "../../datasets/RegDB",
        help="root directory of the dataset",
    )

    parser.add_argument(
        "--save-root",
        type=str,
        default="./check_points",
        help="root directory to save",
    )

    parser.add_argument(
        "--save-prefix",
        type=str,
        default="domain_reversal",
        help="prefix to append with the save folder name",
    )
    
    parser.add_argument("--desc", type=str, default="features", help="Description for save file")
    parser.add_argument(
        "--print-freq", type=float, default=32, help="number of batch to print after"
    )
    parser.add_argument("--trial", type=int, default=1, help="trial")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument(
        "--save-freq", type=float, default=5, help="number of epochs to save after"
    )
    parser.add_argument(
        "--test-freq", type=float, default=5, help="number of epochs to test after"
    )
    parser.add_argument("--margin", type=float, default=0.3, help="triplet loss margin")
    parser.add_argument("--cuda", action="store_true", help="use GPU computation")
    # Testing arguments
    parser.add_argument("--t-to-v", action="store_true", help="If true, Thermal(gallery) to visible (query)")
    # reranking arguments
    parser.add_argument("--rerank", action="store_true", help="use re-ranking")
    parser.add_argument("--k1", type=int, default=20, help="reranking k1")
    parser.add_argument("--k2", type=int, default=4, help="reranking k2")
    parser.add_argument("--rerank-lambda", type=float, default=0.3, help="reranking lambda")

    parser.add_argument("--evaluate", action="store_true", help="only evaluate")
    parser.add_argument(
        "--pretrained-model", type=str, default="", help="pretrained model path"
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="resnetmid",
        help="model architecture (resnet50, pcb_p6, pcb_p4, senet_p6, resnetmid)",
    )

    # sampler arguments
    parser.add_argument(
        "--num-instances", type=int, default=2, help="number of instances per pid"
    )
    
    # loss function lambdas
    parser.add_argument(
        "--lambda-triplet", type=float, default=10, help="triplet loss lambda"
    )
    parser.add_argument(
        "--lambda-domain", type=float, default=1, help="domain loss lambda"
    )
    parser.add_argument(
        "--lambda-attribute", type=float, default=1, help="attribute loss lambda"
    )
    
    parser.add_argument(
        "--lambda-person-id",
        type=float,
        default=1,
        help="lambda for cross entropy person id",
    )

    # scheduler params
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )

    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler cosine/tanh/step (default: "step"',
    )

    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epoch interval to decay LR",
    )

    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=3e-4,
        metavar="LR",
        help="warmup learning rate (default: 0.0001)",
    )

    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )

    return parser.parse_args()


def save_obj(obj, name):
    with open(name + ".pkl", "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count == 0:
            self.count = 1e-6
        self.avg = self.sum / self.count



def load_pretrained_weights(model, weight_path):
    """Load pretrianed weights to model

    Incompatible layers (unmatched in name or size) will be ignored

    Args:
    - model (nn.Module): network model, which must not be nn.DataParallel
    - weight_path (str): path to pretrained weights
    """
    checkpoint = torch.load(weight_path)
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []
    for k, v in state_dict.items():
        # If the pretrained state_dict was saved as nn.DataParallel,
        # keys would contain "module.", which should be ignored.
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)
    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, please check the key names manually (** ignored and continue **)'.format(
                weight_path
            )
        )
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print(
                "** The following layers are discarded due to unmatched keys or layer size: {}".format(
                    discarded_layers
                )
            )
    
    return model

class AccuracyMeter:
    def __init__(self,):
        self.correct = 0
        self.length = 0
    def reset(self,):
        self.correct = 0
        self.length = 0
    def update(self, logits, targets):
        _, y = torch.max(logits, 1)
        self.correct += (targets.long() == y.long()).sum().float()
        self.length += len(y)
    def accuracy(self,):
        accuracy = self.correct / self.length
        self.reset()
        return accuracy.item()

class AttributeAccuracyMeter:
    def __init__(
        self,
        attribute_list,
        attributes2index,
        num_meters,
        exclusion_list=["id", "domain"],
        disable_meter = False,
    ):
        self.acc_meter = {}
        self.attributes2index = attributes2index
        self.attribute_list = attribute_list
        for key in attribute_list:
            if key not in exclusion_list:
                self.acc_meter[key] = AccuracyMeter()
        self.disable_meter = disable_meter

    def reset(self,):
        for key in self.acc_meter:
            self.acc_meter[key].reset()

    def update(self, logits, targets, exclusion_list=["id", "domain"]):
        if not self.disable_meter:
            for key in logits:
                if key not in exclusion_list:
                    self.acc_meter[key].update(
                        logits[key], targets[:, self.attributes2index[key]]
                    )

    def accuracy(self,):
        mean_accuracy = 0.0
        if self.disable_meter:
            return mean_accuracy
        for key in self.acc_meter:
            mean_accuracy += (self.acc_meter[key].accuracy()) / len(self.acc_meter)
        self.reset()
        return mean_accuracy

