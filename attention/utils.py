__author__ = 'yuhongliang324'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss(img_path, costs_train, costs_val):
    plt.figure(figsize=(20, 8))
    plt.plot(costs_train, label='Training Loss')
    plt.plot(costs_val, label='Validation Loss')
    plt.legend()
    plt.savefig(img_path)
