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
    p.add_argument('-lr', '--learning-rate', dest='lr', help='The learning rate given to the optimizer.', type=float, required=False)
    p.add_argument('-lo', '--loss', dest='loss', help='The loss to use during training.', type=str, required=False)
    p.add_argument('-met', '--metrics', dest='metrics', help='The metrics you want to consider during training.', nargs='+', required=False)
    p.add_argument('-trs', '--train-steps', dest='train_steps', help='The number of steps before declaring an epoch finished.', type=int)
    p.add_argument('-vals', '--validation-steps', dest='val_steps', help='The number of steps to draw before stopping during validation.', type=int)
    p.add_argument('-ep', '--epochs', dest='epochs', help='The number of epochs before stopping the training.', type=int)

    args = p.parse_args()
    param = dict(args._get_kwargs())

    if args.train:
        mode = 'new_train' 
    else:
        mode = 'resume_train'
        del param['resume']
    del param['train']
    # Filter removing the values that are None
    param = dict(filter(lambda item: item[1] is not None, param.items()))
    return mode, args.resume, param


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
            plt.savefig(f"../images/plot_{m}.png", facecolor="white")
        plt.show()

