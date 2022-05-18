import argparse
import math
from sklearn.feature_extraction import img_to_graph
import torch
import random
import numpy as np
import os
import cv2

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet18, vgg16
from torchvision import transforms
from tqdm import tqdm
from einops import rearrange
from utils import red_transparent_blue


def parse_args():
    parser = argparse.ArgumentParser("Shapley value visualization", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--ptm", type=str, default="resnet18", choices=["resnet18", "vgg16"],
        help="Pre-trained model")

    parser.add_argument("--input", type=str, default="/Users/zwhe/ALLPhd/课程资料/机器学习/proj/dog.jpg",
        help="Path to input figure.")

    parser.add_argument("--ground-truth", type=str, default="Samoyed",
        help="Ground truth of input images. Select it from imagenet_classes.txt.")

    parser.add_argument("--labels", type=str, default="imagenet_classes.txt", 
        help="All possible labels of imagenet "
        "(https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)."
    )

    parser.add_argument("-m", "--max-context", type=int, default=30, 
        help="Sample N contexts for each input variable."
    )

    parser.add_argument("-lp", "--patch-side-len", type=int, default=28, 
        help="Side length of a patch. Input image shape is (224, 224)."
    )

    return parser.parse_args()

def main(args):
    image_shape = (224, 224)

    # read all categories
    assert os.path.exists(args.labels), "Please download https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

    with open(args.labels, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    assert args.ground_truth in categories, f"{args.ground_truth} not in categories"
    ground_truth_id = categories.index(args.ground_truth)

    # build model
    if args.ptm == "resnet18":
        model = resnet18(pretrained=True, progress=False).eval()
    elif args.ptm == "vgg16":
        model = vgg16(pretrained=True, progress=False).eval()

    # define preprossing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # read input image & perform pre-processing without normalization
    input_img = Image.open(args.input)
    input_tensor = preprocess(input_img).unsqueeze(0)

    fmaps = []
    def forward_hook(module, input, output):
        fmaps.append(output)

    grads = []
    def backward_hook(module, grad_in, grad_out):
        grads.append(grad_out[0].detach())

    getattr(model, "layer4").register_forward_hook(forward_hook)
    getattr(model, "layer4").register_backward_hook(backward_hook)

    output = model(input_tensor)
    idx = np.argmax(output.cpu().data.numpy())

    # backward
    model.zero_grad()
    idx = idx[np.newaxis, np.newaxis]
    idx = torch.from_numpy(idx)
    one_hot = torch.zeros(1, 1000).scatter_(1, idx, 1)
    one_hot.requires_grad = True
    loss = torch.sum(one_hot * output)
    loss.backward()

    # generate CAM
    grads_val = grads[0].cpu().data.numpy().squeeze()
    fmap = fmaps[0].cpu().data.numpy().squeeze()

    cam = np.zeros(fmap.shape[1:], dtype=np.float32)
    alpha = np.mean(grads_val, axis=(1, 2))  # GAP
    for k, ak in enumerate(alpha):
        cam += ak * fmap[k]  # linear combination
    
    cam = np.maximum(cam, 0)  # relu
    cam = cv2.resize(cam, image_shape)
    cam = (cam - np.min(cam)) / np.max(cam)
    
    # show
    # cam_show = cv2.resize(cam, None)
    img_show = np.array(input_img).astype(np.float32) / 255
    cam_show = cv2.resize(cam, (img_show.shape[1], img_show.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_show), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img_show)
    cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    plt.imshow(cam[:, :, ::-1])
    plt.show()
    



if __name__ == "__main__":
    args = parse_args()
    main(args)