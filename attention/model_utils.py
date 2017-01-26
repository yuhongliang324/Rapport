__author__ = 'yuhongliang324'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss(img_path, costs_train, costs_val, accs_train=None, accs_val=None):
    plt.figure(figsize=(10, 8))
    plt.plot(costs_train, label='Training Loss')
    plt.plot(costs_val, label='Validation Loss')
    if accs_train is not None:
        plt.plot(accs_train, label='Training Accuracy')
        plt.plot(accs_val, label='Validation Accuracy')
    plt.legend()
    plt.savefig(img_path)
