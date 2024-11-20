from simplenet import SimpleNet
import torch
import backbones
import utils


backbone = backbones.load("wideresnet50")
layers_to_extract = ["layer2", "layer3"]
device = utils.set_torch_device([0])

