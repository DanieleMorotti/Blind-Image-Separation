import os
import argparse
import matplotlib.pyplot as plt

# Check if the file with the weights exists and if it has the correct extension
def check_file(parser, path):
    ext = os.path.splitext(path)[-1].lower()

    if ext not in [".hdf5", ".h5"] :
        parser.error(f"The file has not the correct extension: {ext} must be .h5 or .hdf5.")
    else:
        if os.path.exists(path):
            return path
        else:
            parser.error(f"The file with the weights doesn't exist in the current path: {path}.")


# Check the inputs of the training script
def check_training_args():
    p = argparse.ArgumentParser()
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('-resume', '--resume-training', dest='resume', help='Resume the training starting from the weights of the prompted file.', type=lambda x: check_file(p, x))
    mode.add_argument('-train', '--new-training', dest='train', help='It starts a new training', action='store_true')

    args = p.parse_args()
    mode = 'new_train' if args.train else 'resume_train'

    return mode, args.resume


# Plot the history considering the evaluated metrics
def plot_history(history, metrics, save_images=False):
    validation = False
    for m in metrics:
        if f'val_{m}' in history.keys():
            validation = True
        plt.plot(history[m], c='b', label=f'train {m}')
        if validation:
            plt.plot(history[f'val_{m}'], c='r', label=f'validation {m}')
        plt.legend(loc='upper right') 
        if save_images:
            plt.savefig(f"./evaluation/images/plot_{m}.png", facecolor="white")
        plt.show()

