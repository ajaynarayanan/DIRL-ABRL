from __future__ import absolute_import
from __future__ import division

import torch, sys
import os.path as osp
from torch import nn
from torch.nn import functional as F
import torchvision
import torch.utils.model_zoo as model_zoo
from .senet import *

__all__ = ["senet_pcb_p6", "senet_pcb_p4"]

def set_stride(module, stride):
    for internal_module in module.modules():
        if isinstance(internal_module, nn.Conv2d) or isinstance(internal_module, nn.MaxPool2d):
            internal_module.stride = stride
    
    return internal_module

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class DimReduceLayer(nn.Module):
    def __init__(self, in_channels, out_channels, nonlinear):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        )
        layers.append(nn.BatchNorm2d(out_channels))

        if nonlinear == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif nonlinear == "leakyrelu":
            layers.append(nn.LeakyReLU(0.1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SENet_PCB(nn.Module):
    """
    Part-based Convolutional Baseline

    Reference:
    Sun et al. Beyond Part Models: Person Retrieval with Refined
    Part Pooling (and A Strong Convolutional Baseline). ECCV 2018.
    """

    def __init__(
        self,
        num_classes,
        loss,
        senet_type,
        parts=6,
        reduced_dim=256,
        nonlinear="relu",
        pretrained=True,
        last_stride=1,
        attribute_list=None,
        **kwargs
    ):
        super().__init__()
        self.loss = loss
        self.parts = parts

        print("instantiating " + self.__class__.__name__ + ", senet_type " + senet_type)

        # explicitly specify loss={'xent'} to get only feature vectors
        self.senet = globals()[senet_type](
            num_classes, loss={"xent"}, pretrained=pretrained
        )  # loss function default is xent

        self.senet.classifier = Identity()
        set_stride(self.senet.layer4, last_stride)

        # pcb layers
        self.parts_avgpool = nn.AdaptiveAvgPool2d((self.parts, 1))
        self.dropout = nn.Dropout(p=0.5)
        self.conv5 = DimReduceLayer(
            self.senet.feature_dim, reduced_dim, nonlinear="relu"
        )

        # part specific classifiers
        self.feature_dim = reduced_dim
        self.classifier = nn.ModuleList(
            [nn.Linear(self.feature_dim, num_classes) for _ in range(self.parts)]
        )
        self.feature_dim = 2048 * self.parts
        
        # Attribute classifier
        self.classifiers = nn.ModuleDict()
        if attribute_list is not None:
            for atrribute_name, choices in attribute_list.items():
                self.classifiers[atrribute_name] = nn.Linear(self.feature_dim, len(choices))
            
    def forward(self, x):
        f = self.senet.featuremaps(x)
        v_g = self.parts_avgpool(f)  # b x c x self.parts x 1

        '''
        if not self.training:
            v_g = F.normalize(v_g, p=2, dim=1)
            return v_g.view(v_g.size(0), -1)
        '''
        v_g = self.dropout(v_g)
        v_h = self.conv5(v_g)

        y = []
        for i in range(self.parts):
            v_h_i = v_h[:, :, i, :]
            v_h_i = v_h_i.view(v_h_i.size(0), -1)
            y_i = self.classifier[i](v_h_i)
            y.append(y_i)

        if self.loss == {"xent"}:
            return y
        elif self.loss == {"xent", "htri"}:
            v_g = F.normalize(v_g, p=2, dim=1)
            fc_features = v_g.view(v_g.size(0), -1)
            classifier_logits = {}
            classifier_logits["id"] = y
            for label, classifier in self.classifiers.items():
                classifier_logits[label] = classifier(fc_features)
            return fc_features, classifier_logits
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


def senet_pcb_p6(
    num_classes,
    loss={"xent", "htri"},
    senet_type="se_resnet50",
    pretrained=True,
    **kwargs
):
    model = SENet_PCB(
        num_classes=num_classes,
        loss=loss,
        last_stride=1,
        parts=6,
        reduced_dim=256,
        senet_type=senet_type,
        pretrained=pretrained,
        **kwargs
    )

    return model


def senet_pcb_p4(
    num_classes,
    loss={"xent", "htri"},
    senet_type="se_resnet50",
    pretrained=True,
    **kwargs
):
    model = SENet_PCB(
        num_classes=num_classes,
        loss=loss,
        last_stride=1,
        parts=4,
        reduced_dim=256,
        senet_type=senet_type,
        pretrained=pretrained,
        **kwargs
    )

    return model
