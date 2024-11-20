from simplenet import SimpleNet
import torch
import backbones
import utils
import PIL
from torchvision import transforms


backbone = backbones.load("wideresnet50")
layers_to_extract = ["layer2", "layer3"]
device = utils.set_torch_device([0])
input_size = (3, 288, 288)
pretrain_dim = 1536
target_dim = 1536

model = SimpleNet(device=device)
model.load(
    backbone=backbone,
    layers_to_extract_from=layers_to_extract,
    device=device,
    input_shape=input_size,
    pretrain_embed_dimension=pretrain_dim,
    target_embed_dimension=target_dim,
)


img_path = "/content/dataset/screw/test/scratch_neck/000.png"
img = PIL.Image.open(img_path).convert("RGB")
# to batch of tensors
transform_img = [transforms.Resize(288),transforms.ToTensor(),]
transform_img = transforms.Compose(transform_img)
img = transform_img(img)

#make batch of 1
img = img.unsqueeze(0)

print(model.test(test_data=img, save_segmentation_images=True, ckpt_path="/content/ckpt.pth"))

