from .resnet import ee_resnet18, resnet18
from .vit import ee_vit_tiny

MODELS = {"resnet18": resnet18, "ee_resnet18": ee_resnet18, "ee_vit_tiny": ee_vit_tiny}


def build_model(name, num_classes=100):
    return MODELS[name](num_classes)
