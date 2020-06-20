import os, sys, pickle
import torch, os.path as osp
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data.sampler import Sampler
import torch.utils.data as data


num_options_per_attributes = {
    "age": ["young", "teen", "adult", "old"],
    "backpack": ["no", "yes"],
    "bag": ["no", "yes"],
    "boots": ["no", "yes"],
    "clothes": ["dress", "pants"],
    "down": ["long", "short"],
    "downcloth": [
        "downblack",
        "downblue",
        "downbrown",
        "downgray",
        "downgreen",
        "downpink",
        "downpurple",
        "downred",
        "downwhite",
        "downyellow",
    ],
    "gender": ["male", "female"],
    "hair": ["short", "long"],
    "handbag": ["no", "yes"],
    "hat": ["no", "yes"],
    "shoes": ["dark", "light"],
    "up": ["long", "short"],
    "upcloth": [
        "upblack",
        "upblue",
        "upbrown",
        "upgray",
        "upgreen",
        "uppurple",
        "upred",
        "upwhite",
        "upyellow",
    ],
}


def get_file_name(filepath):
    return os.path.basename(filepath).split(".")[0]


def load_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def read_ids(root_path, split_type):
    config_file_path = os.path.join(root_path, "exp", split_type + "_id.txt")
    with open(config_file_path, "r") as file:
        file_lines = file.readlines()

    # the file has only one line with ids
    id_line = file_lines[0]
    all_ids = ["%04d" % int(i) for i in id_line.split(",")]
    print(config_file_path + " : " + str(len(all_ids)))
    return all_ids


class cam_ID_folder:
    def __init__(self, root_path, cam_name, ID, cam_config):
        # init the instance variables
        self.root_path = root_path
        self.cam_name = cam_name
        self.ID = ID
        self.folder_path = os.path.join(self.root_path, self.cam_name, self.ID)
        self.cam_config = cam_config

    def is_exists(self):
        # returns true if the folder exists
        return os.path.exists(self.folder_path)

    def get_file_instances(self):
        instances = []

        # if the folder exists
        if self.is_exists():
            print(
                self.folder_path
                + " : "
                + str(len(os.listdir(self.folder_path)))
                + " files"
            )
            # for each of the file in the directory
            for file in os.listdir(self.folder_path):
                # read the file and store it in the list
                filepath = os.path.join(self.folder_path, file)
                img = osp.abspath(filepath)

                instances.append(
                    (img, self.ID, self.cam_config[self.cam_name], self.cam_name)
                )

        return instances


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
            "gender",
            "hair",
            "handbag",
            "hat",
            "shoes",
            "up",
            "upcloth",
            "downcloth",
        ]
        self.num_options_per_attributes = {
            "age": ["young", "teen", "adult", "old"],
            "backpack": ["no", "yes"],
            "bag": ["no", "yes"],
            "boots": ["no", "yes"],
            "clothes": ["dress", "pants"],
            "down": ["long", "short"],
            "downcloth": [
                "downblack",
                "downblue",
                "downbrown",
                "downgray",
                "downgreen",
                "downpink",
                "downpurple",
                "downred",
                "downwhite",
                "downyellow",
            ],
            "gender": ["male", "female"],
            "hair": ["short", "long"],
            "handbag": ["no", "yes"],
            "hat": ["no", "yes"],
            "shoes": ["dark", "light"],
            "up": ["long", "short"],
            "upcloth": [
                "upblack",
                "upblue",
                "upbrown",
                "upgray",
                "upgreen",
                "uppurple",
                "upred",
                "upwhite",
                "upyellow",
            ],
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
    def __init__(self, root_path, IDs, config_name, transforms, attr_mode=None):
        self.root_path = root_path
        self.cam_config = {
            "cam1": "rgb",
            "cam2": "rgb",
            "cam3": "IR",
            "cam4": "rgb",
            "cam5": "rgb",
            "cam6": "IR",
        }

        self.IDs = IDs
        self.config_name = config_name
        self.data_instances = self.read_data_instances()
        self.IDs2Classes = {}
        self.transforms = transforms
        self.attr_mode = attr_mode
        # Attribute information available useful only on training/validation dataset
        # attr_mode can be "train_val", "train", "val"
        if self.attr_mode is not None:
            self.all_ids_attributes = Attributes_IDs(self.root_path, self.attr_mode)

        for index, id in enumerate(self.IDs):
            self.IDs2Classes[id] = index
        
        self.category_type = {"rgb": 0, "IR": 1}
        self.category2string = {0:"rgb", 1:"IR"}    

    def read_data_instances(self):
        data_instances = []

        # check if the config already exists
        config_file = os.path.join(self.root_path, self.config_name + "_config.pth")

        # check if the config file is already existing
        if os.path.exists(config_file):
            # load the existing config file
            print("existing config file " + config_file + " found!. Reading the file!")
            data_instances = torch.load(config_file)
        else:
            # for each of the ids
            for ID in self.IDs:
                # for each of the cameras
                for cam_name in self.cam_config.keys():
                    # get all the data instances
                    folder = cam_ID_folder(
                        self.root_path, cam_name, ID, self.cam_config
                    )
                    data_instances += folder.get_file_instances()

            # save the configuration
            torch.save(data_instances, config_file)

        return data_instances

    def __len__(self):
        return len(self.data_instances)

    def __getitem__(self, index):
        img_path, ID, category, _ = self.data_instances[index]
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)

        # Read attributes for the current ID
        if self.attr_mode is not None:
            attributes = self.all_ids_attributes.determine_attributes(ID)
            return img, self.IDs2Classes[ID], attributes, self.category_type[category]

        return img, self.IDs2Classes[ID], self.category_type[category]


class TestDataset(Dataset):
    def __init__(self, root_path, IDs, config_name, transforms):
        # init the super class
        super().__init__(root_path, IDs, config_name, transforms)

    def read_data_instances(self):
        print("child class method")
        data_instances = {}
        for cam_name in self.cam_config.keys():
            data_instances[cam_name] = {}

        # check if the config already exists
        config_file = os.path.join(self.root_path, self.config_name + "_config.pth")

        # check if the config file is already existing
        if os.path.exists(config_file):
            # load the existing config file
            print("existing config file " + config_file + " found!. Reading the file!")
            data_instances = torch.load(config_file)
        else:
            # for each of the ids
            for ID in self.IDs:
                # for each of the cameras
                for cam_name in self.cam_config.keys():
                    # get all the data instances
                    folder = cam_ID_folder(
                        self.root_path, cam_name, ID, self.cam_config
                    )
                    current_folder_instances = folder.get_file_instances()
                    current_folder_instances = self.order_file_names(
                        current_folder_instances
                    )
                    data_instances[cam_name][ID] = current_folder_instances

            # save the configuration
            torch.save(data_instances, config_file)

        return data_instances

    def get_cam_files_config(self):
        return self.data_instances

    def read_image_from_config(self, config):
        # config will contain
        # img path, ID, category(rgb or IR), cam name
        img_path, _, category, _ = config
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(img)
        return img, category

    def order_file_names(self, instances):
        # create a hash with file name
        filenames_hash = {}
        for inst in instances:
            filename = get_file_name(inst[0])
            # print(filename)
            filenames_hash[filename] = inst

        # create an array ordered by filename, in numerical order
        total_files = len(instances)
        ordered_instances = []
        for i in range(total_files):
            ordered_instances.append(filenames_hash["%04d" % (i + 1)])

        return ordered_instances


def get_dataset(root, split_type, transforms):
    # split_type is a list of strings (dataset types)
    all_ids = []
    # Read the ids given split_type
    for dtype in split_type:
        all_ids += read_ids(root, dtype)
    # Create dataset based on the ids
    attr_mode = None if ("test" in split_type) else "_".join(split_type)
    dataset = Dataset(
        root,
        all_ids,
        config_name="_".join(split_type),
        transforms=transforms,
        attr_mode=attr_mode,
    )
    return dataset


# train_transforms = transforms.Compose(
#     [
#         transforms.Resize((256, 128), Image.BICUBIC),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#     ]
# )
# dataset = get_dataset("../../datasets/SYSU-MM01", ["test"], train_transforms)
# print(dataset[0])
