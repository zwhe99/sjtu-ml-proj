import argparse
import torch
import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet18, vgg16
from utils import jet_transparent
from model.grad_cam import GradCAM
from model.utils import preprocess


def parse_args():
    parser = argparse.ArgumentParser(
        "Grad-CAM visualization", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--ptm", type=str, default="resnet18", choices=["resnet18", "vgg16"],
                        help="Pre-trained model")

    parser.add_argument("--input", type=str, required=True,
                        help="Path to input figure.")

    parser.add_argument("--labels", type=str, default="imagenet_classes.txt",
                        help="All possible labels of imagenet "
                        "(https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)."
                        )

    parser.add_argument("-k", "--top-k", type=int, default=3,
                        help="Compute Grad-CAM for top-k labels"
                        )

    return parser.parse_args()


def main(args):
    # read all categories
    assert os.path.exists(
        args.labels), "Please download https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with open(args.labels, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # build model
    if args.ptm == "resnet18":
        model = resnet18(pretrained=True, progress=False).eval()
    elif args.ptm == "vgg16":
        model = vgg16(pretrained=True, progress=False).eval()

    # read input image & perform pre-processing without normalization
    input_image = Image.open(args.input)

    explainer = GradCAM(model, args.top_k)
    grad_cam_imgs, target_ids = explainer.explain(input_image)

    # display
    fig_size = np.array([5 * args.top_k, 5])
    fig, axes = plt.subplots(nrows=1, ncols=1 + args.top_k, figsize=fig_size)

    # original img
    input_image = preprocess(input_image).permute(1, 2, 0)

    for i in range(1 + args.top_k):
        if i == 0:
            # plot original img
            axes[0].imshow(input_image, cmap=plt.get_cmap('gray'))
            axes[0].set_title("Input")
            axes[0].axis('off')
        else:
            # plot original img as background
            axes[i].imshow(input_image)

            # plot grad cam
            max_val = np.nanpercentile(torch.abs(grad_cam_imgs), 99.9)
            min_val = 0
            im = axes[i].imshow(grad_cam_imgs[i-1],
                                cmap=jet_transparent, vmin=min_val, vmax=max_val)
            axes[i].set_title(categories[target_ids[i-1]])
            axes[i].axis('off')

    # add color bar and show
    fig.subplots_adjust(
        left=0.01,
        right=0.90,
        wspace=0.015
    )

    cax = fig.add_axes([axes[-1].get_position().x1 + 0.005,
                        axes[-1].get_position().y0, 0.02, axes[-1].get_position().height])
    cb = fig.colorbar(im, cax=cax, label="Grad CAM")
    cb.outline.set_visible(False)

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)