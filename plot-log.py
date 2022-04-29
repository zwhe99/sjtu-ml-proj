import argparse
import os
import seaborn as sns
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser("Parse log file and plot training curves", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--log-file", type=str, 
        help="Path to log file")

    parser.add_argument("--out-dir", type=str, default="./imgs",
        help="Path to images directory")

    parser.add_argument("--prefix", type=str, 
        help="Prefix prepend to file name")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    train_stat = {
        'Epoch': [],
        'Step': [],
        'Training_loss': []
    }
    valid_stat = {
        'Step': [],
        'Valid_loss': [],
        'Valid_accuracy': []
    }

    with open(args.log_file, 'r') as f:
        for line in f:
            line = line.split("-")[-1].strip()
            if "[TRAIN]" in line:
                stat = train_stat
            elif "[VALID]" in line:
                stat = valid_stat
            else:
                continue

            for term in line.split(' ')[1:]:
                k, v = term.split('=')
                if k in ['Total_step', 'Step'] :
                    k = "Step"
                    v = int(v)
                
                if k == "Step":
                    v = int(v)

                if "loss" in k or "accuracy" in k:
                    v = float(v)

                stat[k].append(float(v))

    
    sns.set_theme(style="darkgrid")

    # Plot the responses for different events and regions
    plt.figure()
    plot = sns.lineplot(x="Step", y="Training_loss",
                data=train_stat)
    fig = plot.get_figure()
    fig.savefig(f"{args.out_dir}/{args.prefix}-train-loss.png") 

    plt.figure()
    plot = sns.lineplot(x="Step", y="Valid_loss", marker="<",
                data=valid_stat)
    fig = plot.get_figure()
    fig.savefig(f"{args.out_dir}/{args.prefix}-valid-loss.png") 

    plt.figure()
    plot = sns.lineplot(x="Step", y="Valid_accuracy",marker="o",
                data=valid_stat)
    fig = plot.get_figure()
    fig.savefig(f"{args.out_dir}/{args.prefix}-valid-acc.png") 