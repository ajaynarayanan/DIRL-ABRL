import torch, os, sys, torch.nn as nn
import numpy as np
import os.path as osp
from .resnet import resnet50

def set_stride(module, stride):
    for internal_module in module.modules():
        if isinstance(internal_module, nn.Conv2d) or isinstance(internal_module, nn.MaxPool2d):
            internal_module.stride = stride
    
    return internal_module


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        num_classes=1000,
        in_chans=3,
        checkpoint_path="",
        drop_rate=0.3,
        fc_dims=[512],
        attribute_list=None,
    ):
        super().__init__()
        print(
            "instantiating model {} with {} classes, {} in-channels".format(
                model_name, num_classes, in_chans
            )
        )

        self.resnet = resnet50(
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=in_chans,
        )

        # set dilution of layer4
        set_stride(self.resnet.layer4, 1)

        # init fc layer
        num_last_layer_feats = self.resnet.fc.in_features
        self.resnet.fc = self._construct_fc_layer(
            fc_dims, num_last_layer_feats, dropout_p=drop_rate
        )

        # modify the final layer to contain classifiers for person id, attributes
        # attribute_list is expected to contain a dict with key as attribute name and value as array of possible values
        self.classifiers = nn.ModuleDict()
        self.classifiers["id"] = nn.Linear(self.feature_dim, num_classes)
        if attribute_list is not None:
            for atrribute_name, choices in attribute_list.items():
                self.classifiers[atrribute_name] = nn.Linear(self.feature_dim, len(choices))

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer

        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)

    def forward(self, x):
        conv_features = self.resnet.forward_features(x, pool=True)
        fc_features = self.resnet.fc(conv_features)
        classifier_logits = {}
        for label, classifier in self.classifiers.items():
            classifier_logits[label] = classifier(fc_features)

        return fc_features, classifier_logits
