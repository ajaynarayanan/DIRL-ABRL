from .helpers import *
from .adaptive_avgmax_pool import *
from .resnet import *
from .resnet50_model import *
from .pcb_model import *
from .senet_pcb import *
from .resnetmid import *
from .resnetmid_reverse import *
from .resnetmid_twostream import * 
from .osnet_ain import *

models = {
    "resnet50": ResNet,
    "pcb_p6": pcb_p6,
    "pcb_p4": pcb_p4,
    "identity": Identity,
    "senet_p6": senet_pcb_p6,
    "resnetmid" : resnet50mid,
    "resnetmid_domain" : resnet50mid_reverse,
    "resnetmid_twostream":  resnet50mid_twostream,
    "osnet_ain": osnet_ain_x1_0, 
}


def create_reid_model(model_name, **kwargs):
    assert model_name in models, "unknown model name " + model_name
    if model_name in models:
        model = models[model_name](**kwargs)

    return model