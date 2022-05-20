import argparse
import enum
import torch
import sys
import logging
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model.pca import PCA
from model.tsne import TSNE
from model.mnist_classifier import MNISTClassifier
from trainer import Trainer
from utils import eval_helper, seed_everything

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

NUM_CLASSES = 10

def parse_args():
    parser = argparse.ArgumentParser("MNIST training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Device
    parser.add_argument("--cuda", type=bool, default=True, 
        help="Use GPU for training. If not available, degenerate to CPU.")

    # Data
    parser.add_argument("--data-dir", type=str, default="./data", 
        help="Data directory to store SST2 dataset")
    
    # Model parameters
    parser.add_argument("--restore-file", type=str, 
        help="Filename from which to load checkpoint")

    # Training
    parser.add_argument("-bs", "--batch-size", type=int, default=16, 
        help="Batch size")
    parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3, 
        help="Learning rate")
    parser.add_argument("--max-epoch", type=int, default=2, 
        help="Force stop training at specified epoch")
    parser.add_argument("--max-update", type=int, default=5000, 
        help="Force stop training at specified update")
    parser.add_argument("--log-interval", type=int, default=10,
        help="Log every N updates")
    parser.add_argument("--validate-interval-updates", type=int, default=250,
        help="Validate every N updates")
    parser.add_argument("--checkpoint-dir", type=str, default="./ckpts/mnist-classifier",
        help="Path to save checkpoints")

    # Predict
    parser.add_argument("--predict-only", default=False, action="store_true", 
        help="Predict test set and quit. Test set and valid set are the same for MNIST.")
    parser.add_argument("--out-file", type=str, default="pred.tsv",  
        help="Path to prediction file")

    # Eval
    parser.add_argument("--eval-only", default=False, action="store_true", 
        help="Evaluate model on valid set, and plot features using PCA and TSNE. Test set and valid set are the same for MNIST.")

    # Seed for reproducibility
    parser.add_argument("--seed", type=int, default=42, 
        help="Seed for pseudo randomness")

    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    logger.info(args)

    seed_everything(args.seed)

    train_datapipe = MNIST(root=args.data_dir, train=True, transform=transforms.ToTensor(), download=True)
    dev_datapipe = MNIST(root=args.data_dir, train=False, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(train_datapipe, batch_size=args.batch_size)
    dev_dataloader = DataLoader(dev_datapipe, batch_size=args.batch_size)

    # build model
    model = MNISTClassifier()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.restore_file:
        logger.info(f"Loading model from {args.restore_file}")
        model.load_state_dict(torch.load(args.restore_file))

    model.to(device)

    # optimizer
    optim = AdamW(model.parameters(), lr=args.learning_rate)
    criteria = nn.CrossEntropyLoss()

    if (not args.predict_only) and (not args.eval_only):
        # start training
        trainer = Trainer(
            model=model,
            device=device,
            optimizer=optim,
            criteria=criteria,
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_epoch=args.max_epoch,
            max_update=args.max_update,
            validate_interval_updates=args.validate_interval_updates,
            log_interval=args.log_interval,
            checkpoint_dir=args.checkpoint_dir
        )
        trainer.train()
    elif args.predict_only:
            # predict test set

        logger.info(f"[TEST] Predicting valid set ...")
        valid_outputs, valid_features, valid_targets, loss = eval_helper(model, dev_dataloader, criteria)

        with open(args.out_file, 'w') as f:
            f.writelines([str(p) + '\n' for p in valid_outputs])

    elif args.eval_only:
        # evaluate model on valid set, and plot features using PCA and t-SNE
        if not args.restore_file:
            logger.warning(f"[EVAL] You are now evaluating a randomly initialized model."
                "You can set --restore-file to a trained model.")

        logger.info(f"[EVAL] Evaluating model on training set ...")
        train_outputs, train_features, train_targets, train_loss = eval_helper(model, train_dataloader, criteria)

        logger.info(f"[EVAL] Evaluating model on valid set ...")
        valid_outputs, valid_features, valid_targets, valid_loss = eval_helper(model, dev_dataloader, criteria)

        # log
        train_accuracy = (train_outputs == train_targets).sum() / len(train_targets)
        valid_accuracy = (valid_outputs == valid_targets).sum() / len(valid_targets)
        logger.info(f"[EVAL] Train_loss={train_loss:.4f} Train_accuracy={train_accuracy:.4f} Valid_loss={valid_loss:.4f} Valid_accuracy={valid_accuracy:.4f}")

        # feature visualization
        layer_names = ["sub-layer0","sub-layer1","sub-layer2","sub-layer3", "fc1", "fc2"]
        fig_size = np.array([2.5 * len(layer_names), 5])
        fig, axes = plt.subplots(nrows=2, ncols=len(layer_names), figsize=fig_size)
        axes[0, 0].text(
            0, 0.5, 
            'PCA',
            horizontalalignment='right',
            verticalalignment='center',
            rotation='vertical',
            transform=axes[0, 0].transAxes
        )
        axes[1, 0].text(
            0, 0.5, 
            't-SNE',
            horizontalalignment='right',
            verticalalignment='center',
            rotation='vertical',
            transform=axes[1, 0].transAxes
        )

        for j, layer_name in enumerate(layer_names): 
            logger.info(f"[EVAL] Visualize {layer_name} ...")

            valid_outputs, valid_features, valid_targets, valid_loss = eval_helper(model, dev_dataloader, criteria, layer_name=layer_name)
    
            # The valid set of MNIST has 10k samples. We only sample 1k for visualization. 
            rand_idx = np.random.choice(range(len(valid_features)), size=1000, replace=False)
            subset_valid_features = valid_features[rand_idx]
            subset_valid_targets = valid_targets[rand_idx]

            # pca
            logger.info(f"[EVAL] PCA transforming ...")
            pca_features = PCA().fit_transform(subset_valid_features)

            axes[0, j].yaxis.set_visible(False)
            axes[0, j].xaxis.set_visible(False)
            axes[0, j].set_title(layer_name)
            for label in np.unique(subset_valid_targets):
                label_features = pca_features[subset_valid_targets == label]
                axes[0, j].scatter(label_features[:, 0], label_features[:, 1], s=5, label=label)
                

            # tsne
            logger.info(f"[EVAL] t-SNE transforming ...")
            tsne_features = TSNE().fit_transform(subset_valid_features)

            axes[1, j].yaxis.set_visible(False)
            axes[1, j].xaxis.set_visible(False)
            for label in np.unique(subset_valid_targets):
                label_features = tsne_features[subset_valid_targets == label]
                axes[1, j].scatter(label_features[:, 0], label_features[:, 1], s=5, label=label)

        # # set other information
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center right')
        fig.suptitle("Feature visualization of designed model on MNIST test set (1k subset)")
        plt.tight_layout()
        plt.show()