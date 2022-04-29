import argparse
import torch
import torchtext
import math
import sys
import logging
import torch.nn as nn
import torchtext.transforms as T
import torchtext.functional as F
from torchtext.datasets import SST2
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import OrderedDict

from model.lstm import LSTMClassifier
from trainer import Trainer
from utils import pad_sequence, seed_everything

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

NUM_CLASSES = 2
UNK = "<unk>"
PAD = "<pad>"
BOS = "<s>"
EOS = "</s>"


def parse_args():
    parser = argparse.ArgumentParser("LSTM training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Device
    parser.add_argument("--cuda", type=bool, default=True, 
        help="Use GPU for training. If not available, degenerate to CPU.")

    # Data
    parser.add_argument("--data-dir", type=str, default="./data", 
        help="Data directory to store SST2 dataset")
    parser.add_argument("--vocab", type=str, default="./data/SST2/SST-2/lstm.vocab", 
        help="Path to vocabulary")
    parser.add_argument("--spm-model", type=str, default="./data/SST2/SST-2/lstm.model", 
        help="Path to sentencepiece model")
    parser.add_argument("--max-seq-len", type=int, default=256, 
        help="Maximum length of sequences")
    
    # LSTM parameters
    parser.add_argument("--restore-file", type=str, 
        help="Filename from which to load checkpoint")
    parser.add_argument("--emb-dim", type=int, default=512, 
        help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=512, 
        help="Hidden state dimension")
    parser.add_argument("--ff-dim", type=int, default=512, 
        help="Feed forward network intermidiate dimension")

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
    parser.add_argument("--checkpoint-dir", type=str, default="./ckpts/lstm",
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

    # get device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # load the vocab
    with open(args.vocab, 'r') as f:
        vocab = torchtext.vocab.vocab(
            OrderedDict([(line.split('\t')[0], float(line.split('\t')[1])) for line in f]), 
            min_freq = -math.inf,
            specials=[UNK, PAD, BOS, EOS]
        )
    vocab.set_default_index(vocab[UNK])
    padding_idx = vocab[PAD]
    bos_idx = vocab[BOS]
    eos_idx = vocab[EOS]

    # load data
    text_transform = T.Sequential(
        T.SentencePieceTokenizer(args.spm_model),
        T.VocabTransform(vocab),
        T.Truncate(args.max_seq_len - 2),
        T.AddToken(token=bos_idx, begin=True),
        T.AddToken(token=eos_idx, begin=False),
    )

    train_datapipe = SST2(root=args.data_dir, split="train")
    dev_datapipe = SST2(root=args.data_dir, split="dev")
    test_datapipe = SST2(root=args.data_dir, split="test")

    def collate_fn(batch):
        input = F.to_tensor(
                pad_sequence(
                    batch["token_ids"], 
                    left_pad=True, 
                    padding_value=padding_idx
                )
            ).to(device)

        if not args.predict_only:
            target = torch.tensor(batch["target"]).to(device)
        else:
            target = None
        return input, target

    train_datapipe = train_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
    train_datapipe = train_datapipe.batch(args.batch_size)
    train_datapipe = train_datapipe.rows2columnar(["token_ids", "target"])
    train_dataloader = DataLoader(train_datapipe, collate_fn=collate_fn, batch_size=None)

    dev_datapipe = dev_datapipe.map(lambda x: (text_transform(x[0]), x[1]))
    dev_datapipe = dev_datapipe.batch(args.batch_size)
    dev_datapipe = dev_datapipe.rows2columnar(["token_ids", "target"])
    dev_dataloader = DataLoader(dev_datapipe, collate_fn=collate_fn, batch_size=None)

    test_datapipe = test_datapipe.map(lambda x: (text_transform(x[0]), None))
    test_datapipe = test_datapipe.batch(args.batch_size)
    test_datapipe = test_datapipe.rows2columnar(["token_ids", "target"])
    test_dataloader = DataLoader(test_datapipe, collate_fn=collate_fn, batch_size=None)

    # build model
    model = LSTMClassifier(
        emb_dim=args.emb_dim, 
        hidden_dim=args.hidden_dim, 
        ff_dim=args.ff_dim,
        vocab_size=len(vocab), 
        padding_idx=padding_idx, 
        num_classes=NUM_CLASSES
    )
    if args.restore_file:
        logger.info(f"Loading model from {args.restore_file}")
        model.load_state_dict(torch.load(args.restore_file))

    model.to(device)

    if not args.predict_only:
        # do training

        # optimizer
        optim = AdamW(model.parameters(), lr=args.learning_rate)
        criteria = nn.CrossEntropyLoss()

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
        
        model.eval()
        with torch.no_grad():
            for batch in test_dataloader:
                input, _ = batch[0], batch[1]    
                outputs.append(model(input).argmax(1))  

        outputs = torch.concat(outputs).cpu().numpy()

        with open(args.out_file, 'w') as f:
            f.writelines([str(p) + '\n' for p in outputs])

