from .resnet50_model import *
from .resnetmid import *
from .resnetmid_reverse import * 

models = {
    "resnet50": ResNet,
    "identity": Identity,
    "resnetmid" : resnet50mid,
    "resnetmid101" : resnet101mid,
    "resnetmid_domain": resnet50mid_reverse,
}


def create_reid_model(model_name, **kwargs):
    assert model_name in models, "unknown model name " + model_name
    if model_name in models:
        model = models[model_name](**kwargs)

    return model
