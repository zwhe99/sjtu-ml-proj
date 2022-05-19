import argparse
import torch
import numpy as np
import os

from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import resnet18, vgg16
from utils import red_transparent_blue
from model.shapley_value import ShapleyValue
from model.utils import preprocess


def parse_args():
    parser = argparse.ArgumentParser(
        "Shapley value visualization", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--ptm", type=str, default="resnet18", choices=["resnet18", "vgg16"],
                        help="Pre-trained model")

    parser.add_argument("--input", type=str, required=True,
                        help="Path to input figure.")

    parser.add_argument("--labels", type=str, default="imagenet_classes.txt",
                        help="All possible labels of imagenet "
                        "(https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt)."
                        )

    parser.add_argument("-k", "--top-k", type=int, default=3,
                        help="Compute shapley value for top-k labels"
                        )

    parser.add_argument("-m", "--max-context", type=int, default=30,
                        help="Sample N contexts for each input variable."
                        )

    parser.add_argument("-lp", "--patch-side-len", type=int, default=28,
                        help="Side length of a patch. Input image shape is (224, 224)."
                        )

    return parser.parse_args()


def main(args):
    # read all categories
    assert os.path.exists(args.labels), "Please download https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    with open(args.labels, "r") as f:
        categories = [s.strip() for s in f.readlines()]

    # build model
    if args.ptm == "resnet18":
        model = resnet18(pretrained=True, progress=False).eval()
    elif args.ptm == "vgg16":
        model = vgg16(pretrained=True, progress=False).eval()


    # read input image & perform pre-processing without normalization
    input_image = Image.open(args.input)

    explainer = ShapleyValue(model, args.max_context,
                                args.patch_side_len, top_k=args.top_k)
    shaply_value_imgs, target_ids = explainer.explain(input_image)

    # display
    fig_size = np.array([5 * args.top_k, 5])
    fig, axes = plt.subplots(nrows=1, ncols=1 + args.top_k, figsize=fig_size)

    # original img
    input_image = preprocess(input_image).permute(1, 2, 0)

    # gray img
    input_image_gray = (0.2989 * input_image[:, :, 0] + 0.5870 *
                        input_image[:, :, 1] + 0.1140 * input_image[:, :, 2])  # rgb to gray

    for i in range(1 + args.top_k):
        if i == 0:
            # plot original img
            axes[0].imshow(input_image, cmap=plt.get_cmap('gray'))
            axes[0].set_title("Input")
            axes[0].axis('off')
        else:
            # plot shapley values
            max_val = np.nanpercentile(torch.abs(shaply_value_imgs), 99.9)
            im = axes[i].imshow(shaply_value_imgs[i-1],
                                    cmap=red_transparent_blue, vmin=-max_val, vmax=max_val)
            axes[i].set_title(categories[target_ids[i-1]])
            axes[i].axis('off')

            # plot gray img as background
            axes[i].imshow(input_image_gray,
                                cmap=plt.get_cmap('gray'), alpha=0.15)

    # add color bar and show
    fig.subplots_adjust(
        left=0.075,
        right=0.99,
        wspace=0.015
    )

    cax = fig.add_axes([axes[0].get_position().x0 - 0.03,
                        axes[0].get_position().y0, 0.02, axes[0].get_position().height])
    cb = fig.colorbar(im, cax=cax, label="Shapley Value")
    cb.outline.set_visible(False)
    cax.yaxis.tick_left()
    cax.yaxis.set_label_position('left')
    fig.suptitle(f'Shapley Value for {args.ptm} ({os.path.basename(args.input)})') 

    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
