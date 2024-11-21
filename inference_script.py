from simplenet import SimpleNet
import torch
import backbones
import utils
import PIL
from torchvision import transforms

from datasets.mvtec import MVTecDataset
from datasets.mvtec import DatasetSplit
import cv2
import os


test_dataset = MVTecDataset(
                "/content/dataset",
                classname="screw",
                resize=329,
                imagesize=288,
                split=DatasetSplit.TEST,
                seed=0,
)

test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=8,
                shuffle=False,
                num_workers=4,
                prefetch_factor=2,
                pin_memory=True,
            )


backbone = backbones.load("wideresnet50")
layers_to_extract = ["layer2", "layer3"]
device = utils.set_torch_device([0])
input_size = (3, 288, 288)
pretrain_dim = 1536
target_dim = 1536
pre_proj = 1
model = SimpleNet(device=device)
model.load(
    backbone=backbone,
    layers_to_extract_from=layers_to_extract,
    device=device,
    input_shape=input_size,
    pretrain_embed_dimension=pretrain_dim,
    target_embed_dimension=target_dim,
    pre_proj=pre_proj,

)


""" img_path = "/content/dataset/screw/test/scratch_neck/000.png"
img = PIL.Image.open(img_path).convert("RGB")
# to batch of tensors
transform_img = [transforms.Resize(288),transforms.ToTensor(),transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
transform_img = transforms.Compose(transform_img)
img = transform_img(img)

#make batch of 1
img = img.unsqueeze(0) """

state_dict = {}
ckpt_path = "/content/screw_ckpt.pth"
state_dict = torch.load(ckpt_path, map_location=model.device)
if 'discriminator' in state_dict:
    print("Loading discriminator weights")
    model.discriminator.load_state_dict(state_dict['discriminator'])
    if "pre_projection" in state_dict:
        print("Loading pre_projection weights")
        model.pre_projection.load_state_dict(state_dict["pre_projection"])
else:
    print("Loading model weights")
    model.load_state_dict(state_dict, strict=False)


scores, masks, features, labels_gt, masks_gt = model.predict(test_dataloader)


model.save_segmentation_images(test_dataloader, masks, scores)
