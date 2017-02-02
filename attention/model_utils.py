__author__ = 'yuhongliang324'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss(img_path, costs_train, costs_val, dyad, losses_krip_train=None, losses_krip_val=None):
    plt.figure(figsize=(10, 8))
    plt.plot(costs_train, label='Training Loss')
    plt.plot(costs_val, label='Validation Loss')
    if losses_krip_train is not None:
        plt.plot(losses_krip_train, '--', label='Training Krip Loss')
        plt.plot(losses_krip_val, '--', label='Validation Krip Loss')
    plt.legend()
    plt.title('Dyad ' + str(dyad))
    plt.savefig(img_path)
