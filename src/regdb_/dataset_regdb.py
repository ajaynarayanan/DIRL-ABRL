import os, sys
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy, random
import pickle 

def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)

class Attributes_IDs:
    def __init__(self, root_path, file_name):
        # Attribute file name and it's path
        self.file_name = os.path.join(root_path, file_name)
        # Read the attributes for each ID
        self.attribute_source = self.read_attributes()
        # attributes_list is a list of strings where each string is an attribute
        self.attribute_list = [
            "age",
            "backpack",
            "bag",
            "boots",
            "clothes",
            "down",
            "hair",
            "handbag",
            "hat",
            "shoes",
            "up",
            "gender",
            "facing",
        ]
        self.num_options_per_attributes = {
            "age": ["young", "teen", "adult", "old"],
            "backpack": ["no", "yes"],
            "bag": ["no", "yes"],
            "boots": ["no", "yes"],
            "clothes": ["dress", "pants"],
            "down": ["long", "short"],
            "gender": ["male", "female"],
            "hair": ["short", "long"],
            "handbag": ["no", "yes"],
            "hat": ["no", "yes"],
            "shoes": ["dark", "light"],
            "up": ["long", "short"],
            "facing" : ["front", "back"],
        }

        self.attribute2index = {}
        for index, key in enumerate(self.attribute_list):
            self.attribute2index[key] = index

        self.no_attributes = [-1] * len(self.attribute_list)

    def read_attributes(self):
        # Read the dictionary of attributes for the ids
        ids_attributes = load_obj(self.file_name)
        return ids_attributes

    def get_attribute_value(self, attribute_dict, key):
        if key in attribute_dict:
            return attribute_dict[key]
        else:
            # default to know if the attribute is not available
            return -1

    def determine_attributes(self, ID):
        if ID not in self.attribute_source:
            attributes = self.no_attributes
        else:
            current_id_attributes = self.attribute_source[ID]

            # convert the attributes of identity (in dict) to the attribute array based on attribute list
            attributes = []
            for attribute_name in self.attribute_list:
                attribute_value = self.get_attribute_value(
                    current_id_attributes, attribute_name
                )

                attributes.append(attribute_value)

        attributes = np.array(attributes)

        return attributes


class Dataset(data.Dataset):
    category_type = {"rgb": 0, "IR": 1} # thermal = IR (considered)
    index2category = {val:key for key, val in category_type.items()}    
    
    def __init__(self, root_path, file_prefix, trail, attr_mode = None):
        """
        root_path   : Absolute path of the dataset
        file_prefix : Either "train" or "test"
        trail       : Trail index belongs [1, 10]
        attr_mode   : Use of attributes in training process
        """
        self.root_path = root_path
        self.trail = str(trail)
        self.file_prefix = file_prefix

        # define transforms        # 
        if self.file_prefix == 'train':
            self.transform = transforms.Compose(
                [
                    transforms.Resize((256, 128), Image.BICUBIC),
                    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.95, 1.05)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif self.file_prefix == 'test':
            self.transform = transforms.Compose(
                        [
                            transforms.Resize((256, 128), Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ]
                    )            
        else:
            assert False, 'unknown file prefix ' + self.file_prefix
        
        # define file paths
        self.thermal_config = os.path.join(
            root_path, "idx", self.file_prefix + "_thermal_" + self.trail + ".txt"
        )
        self.visible_config = os.path.join(
            root_path, "idx", self.file_prefix + "_visible_" + self.trail + ".txt"
        )
        
        self.index2label = {}
        self.data_instances, self.visible, self.thermal = self.read_data_instances()
        self.attr_mode = attr_mode
        # Attribute information available useful only on training/validation dataset (trail 1)
        if self.attr_mode is not None:
            self.all_ids_attributes = Attributes_IDs(self.root_path, "attrs")

    def number_classes(self,):
        return len(self.index2label)

    def read_from_file(self, file_name, category):
        instances = []
        # Read all the instances from the given file
        content = [line.rstrip("\n") for line in open(file_name)]
        # Split img_path and ID
        instances = [line.split(" ") for line in content]
        # Map training index to true label 
        for item in instances:
            file_path, index = item
            label = file_path.split("/")[1]
            self.index2label[index] = label
        # Add root path to the img path
        for i in range(len(instances)):
            instances[i][0] = os.path.join(self.root_path, instances[i][0])
        # Add category to the data instances
        instances = [line + [category] for line in instances]
        return instances

    def read_data_instances(self):
        visible_instances = self.read_from_file(self.visible_config, 1)
        thermal_instances = self.read_from_file(self.thermal_config, 0)
        data_instances = visible_instances + thermal_instances
        return data_instances, visible_instances, thermal_instances

    def __len__(self):
        return len(self.data_instances)

    def __getitem__(self, index):

        # Read data instances
        img_path, ID, category = self.data_instances[index]
        # Read image, transform the images
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Read attributes for the current ID
        if self.attr_mode is not None:
            attributes = self.all_ids_attributes.determine_attributes(self.index2label[ID])
            return img, int(ID), attributes, category

        return img, int(ID), category



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
        self.num_pids_per_batch = self.batch_size // (2 * self.num_per_domain_instances) # twice the instances due to IR and rgb
        self.index_dic = defaultdict(lambda: {'IR':[], 'rgb':[]})
        
        for index, (_, pid_string, category) in enumerate(self.data_source):
            self.index_dic[pid_string][Dataset.index2category[category]].append(index)
        
        self.pids = list(self.index_dic.keys())
        print('sampler: {} identities found'.format(len(self.pids)))
        print('sampler: {} ids/batch, {} batch size, {} instances per domain'.format(
                                self.num_pids_per_batch, self.batch_size, self.num_per_domain_instances))

        # estimate number of examples in an epoch
        self.length = 0
        self.pid_per_domain_instances = {}
        for pid in self.pids:
            IR_idxs = self.index_dic[pid]['IR']
            rgb_idxs = self.index_dic[pid]['rgb']
            # Use either min or average
            num = min(len(IR_idxs), len(rgb_idxs))
            #num = (len(IR_idxs) + len(rgb_idxs)) // 2
            if num < self.num_per_domain_instances:
                num = self.num_per_domain_instances
            
            # per domain instance count
            self.pid_per_domain_instances[pid] = num - num % self.num_per_domain_instances
            
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
            IR_indices = sample_instances(self.index_dic[pid]['IR'], self.pid_per_domain_instances[pid])
            rgb_indices = sample_instances(self.index_dic[pid]['rgb'], self.pid_per_domain_instances[pid])
                      
            batch_idxs = []
            for IR_idx, rgb_idx in zip(IR_indices, rgb_indices):
                batch_idxs += [IR_idx, rgb_idx] # add both IR and RGB images

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