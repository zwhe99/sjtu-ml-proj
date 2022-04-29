import argparse
import torch
import sys
import logging
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.optim import AdamW
from torch.utils.data import DataLoader

from model.net import Net
from trainer import Trainer
from utils import seed_everything

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
        help="Predict test set and quit")
    parser.add_argument("--out-file", type=str, default="pred.tsv",  
        help="Path to prediction file")

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
    model = Net()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.restore_file:
        logger.info(f"Loading model from {args.restore_file}")
        model.load_state_dict(torch.load(args.restore_file))

    model.to(device)

    # optimizer
    optim = AdamW(model.parameters(), lr=args.learning_rate)
    criteria = nn.CrossEntropyLoss()

    if not args.predict_only:
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
    else:
        # predict test set

        logger.info(f"[TEST] Predicting test set ...")
        outputs = []
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        counter = 0
        
        model.eval()
        with torch.no_grad():
            for batch in dev_dataloader:
                input, target = batch[0], batch[1]   
                output = model(input) 
                loss = criteria(output, target).item()
                total_loss += float(loss)
                correct_predictions += (output.argmax(1) == target).type(torch.float).sum().item()
                total_predictions += len(target)
                counter += 1

                outputs.append(output.argmax(1))  

        # write prediction to output file
        outputs = torch.concat(outputs).cpu().numpy()
        with open(args.out_file, 'w') as f:
            f.writelines([str(p) + '\n' for p in outputs])

        # log acc and loss
        test_loss = total_loss / counter
        test_accuracy = correct_predictions / total_predictions
        logger.info(f"[TEST] Test_loss={test_loss:.5f} Test_accuracy={test_accuracy:.4f}")
        