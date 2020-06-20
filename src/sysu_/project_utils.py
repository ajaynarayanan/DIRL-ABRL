import os, sys, argparse
import numpy as np
import torch, torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os.path as osp, pickle
import copy, random
from collections import defaultdict

from dataset_sysu import *
from my_utils import *


class RandomIdentity_IRRGBSampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_per_domain_instances (int): number of per domain instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_per_domain_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_per_domain_instances = num_per_domain_instances
        self.num_pids_per_batch = self.batch_size // (
            2 * self.num_per_domain_instances
        )  # twice the instances due to IR and rgb
        self.index_dic = defaultdict(lambda: {"IR": [], "rgb": []})

        for index, (_, pid_string, category, _) in enumerate(self.data_source):
            self.index_dic[pid_string][category].append(index)

        self.pids = list(self.index_dic.keys())
        print("sampler: {} identities found".format(len(self.pids)))
        print(
            "sampler: {} ids/batch, {} batch size, {} instances per domain".format(
                self.num_pids_per_batch, self.batch_size, self.num_per_domain_instances
            )
        )

        # estimate number of examples in an epoch
        self.length = 0
        self.pid_per_domain_instances = {}
        for pid in self.pids:
            IR_idxs = self.index_dic[pid]["IR"]
            rgb_idxs = self.index_dic[pid]["rgb"]
            # Use either min or average
            num = min(len(IR_idxs), len(rgb_idxs))
            # num = (len(IR_idxs) + len(rgb_idxs)) // 2
            if num < self.num_per_domain_instances:
                num = self.num_per_domain_instances

            # per domain instance count
            self.pid_per_domain_instances[pid] = (
                num - num % self.num_per_domain_instances
            )

            # twice the num due to IR, rgb images
            self.length += 2 * self.pid_per_domain_instances[pid]

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        def sample_instances(indices, count):
            idxs = copy.deepcopy(indices)
            if len(idxs) < count:
                idxs = np.random.choice(idxs, size=count, replace=True)

            random.shuffle(idxs)
            return idxs

        for pid in self.pids:
            IR_indices = sample_instances(
                self.index_dic[pid]["IR"], self.pid_per_domain_instances[pid]
            )
            rgb_indices = sample_instances(
                self.index_dic[pid]["rgb"], self.pid_per_domain_instances[pid]
            )

            batch_idxs = []
            for IR_idx, rgb_idx in zip(IR_indices, rgb_indices):
                batch_idxs += [IR_idx, rgb_idx]  # add both IR and RGB images

                # if the currently collected batch amounts to 2*num_instances, add them to the buffer
                if len(batch_idxs) == 2 * self.num_per_domain_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length


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


def init_logger(opt):
    log_file_path = osp.join(opt.save_root, "train.log")
    return Logger(fpath=log_file_path)


def get_datasets(opt):

    # specify the rgb, IR transform
    train_transforms = transforms.Compose(
        [
            transforms.Resize((256, 128), Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((256, 128), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    print("Train transforms used : ", train_transforms)
    print("Test transforms used : ", test_transforms)

    
    if opt.test_on_val:
        # Testing is on validation dataset and training is on train dataset
        split_types = ["train"]
        test_config = "val"
    else:
        # Testing is on test dataset and training is on train and validation dataset
        split_types = ["train", "val"]
        test_config = "test"

    # Prepare training dataset with attributes
    train_dataset = get_dataset(opt.sysu_root, split_types, train_transforms)
    attribute_list = train_dataset.all_ids_attributes.attribute_list
    attribute_choices = train_dataset.all_ids_attributes.num_options_per_attributes
    attribute2index = train_dataset.all_ids_attributes.attribute2index
    label2index = train_dataset.IDs2Classes
    
    print(attribute2index, attribute_choices)
    # prepare test dataset
    test_ids = read_ids(opt.sysu_root, test_config)
    test_dataset = TestDataset(
        opt.sysu_root, test_ids, test_config, transforms=test_transforms)

    return (
        train_dataset,
        test_dataset,
        test_ids,
        attribute_list,
        attribute_choices,
        label2index,
        attribute2index,
    )


def OptionsParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of epochs of training"
    )
    parser.add_argument(
        "--train-batch", type=int, default=64, help="size of the batches"
    )

    parser.add_argument(
        "--sysu_root",
        type=str,
        default="../../datasets/SYSU-MM01/",
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
        default="attribute_based_IR",
        help="prefix to append with the save folder name",
    )

    parser.add_argument(
        "--print-freq", type=float, default=100, help="number of batch to print after"
    )
    parser.add_argument(
        "--save-freq", type=float, default=5, help="number of epochs to save after"
    )
    parser.add_argument(
        "--test-freq", type=float, default=5, help="number of epochs to test after"
    )
    parser.add_argument("--test-mode", type=str, default="all", help="all or indoor")
    parser.add_argument(
        "--test-number-shot", type=int, default=1, help="single (1) or multi (10) shot"
    )
    parser.add_argument("--margin", type=float, default=0.3, help="triplet loss margin")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")
    parser.add_argument("--optim", type=str, default="adam", help="sgd/adam")
    parser.add_argument(
        "--cuda", action="store_true", help="use GPU computation"
    )
    parser.add_argument(
        "--test-on-val", action="store_true", help="If true, Testing on validation"
    )
    
    parser.add_argument(
        "--num-instances", type=int, default=2, help="number of instances per pid"
    )
    parser.add_argument("--evaluate", action="store_true", help="only evaluate")
    parser.add_argument(
        "--pretrained-model", type=str, default="", help="pretrained model path"
    )
    parser.add_argument(
        "--preepoch", type=int, default=0, help="pretrained model epoch"
    )

    parser.add_argument(
        "--arch",
        type=str,
        default="resnetmid",
        help="model architecture (resnetmid_domain)",
    )

    # loss function lambdas
    parser.add_argument(
        "--lambda-triplet", type=float, default=1, help="triplet loss lambda"
    )
    parser.add_argument(
        "--gamma-triplet", type=float, default=1, help="gamma for triplet loss lambda"
    )
    parser.add_argument(
        "--lambda-domain", type=float, help="domain loss lambda", default=1
    )
    parser.add_argument(
        "--lambda-attribute", type=float, default=10.0, help="Attribute loss lambda"
    )
    parser.add_argument(
        "--lambda-trip-attribute",
        type=float,
        default=0,
        help="Triplet Attribute loss lambda",
    )

    parser.add_argument(
        "--lambda-person-id",
        type=float,
        default=10,
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
        default="cosine",  # step, cosine
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
        exclusion_list=["id", "upcloth", "downcloth", "domain"],
    ):
        self.acc_meter = {}
        self.attributes2index = attributes2index
        self.attribute_list = attribute_list
        for key in attribute_list:
            if key not in exclusion_list:
                self.acc_meter[key] = AccuracyMeter()

    def reset(self,):
        for key in self.acc_meter:
            self.acc_meter[key].reset()

    def update(self, logits, targets, exclusion_list=["id"]):
        for key in logits:
            if key not in exclusion_list:
                self.acc_meter[key].update(
                    logits[key], targets[:, self.attributes2index[key]]
                )

    def accuracy(self,):
        mean_accuracy = 0.0
        for key in self.acc_meter:
            mean_accuracy += (self.acc_meter[key].accuracy()) / len(self.acc_meter)
        self.reset()
        return mean_accuracy
