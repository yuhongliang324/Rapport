__author__ = 'yuhongliang324'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_loss(img_path, dyad, costs_train, costs_val, losses_krip_train=None, losses_krip_val=None,
              costs_test=None, losses_krip_test=None, tdyad=None):
    plt.figure(figsize=(10, 8))
    plt.plot(costs_train, 'b-', label='Training Loss')
    plt.plot(costs_val, 'm--', label='Validation Loss')
    if losses_krip_train is not None:
        plt.plot(losses_krip_train, label='Training Krip Loss')
        plt.plot(losses_krip_val, '--', label='Validation Krip Loss')
    if costs_test is not None:
        plt.plot(costs_test, 'c--', label='Test Loss')
    if losses_krip_test is not None:
        plt.plot(losses_krip_test, label='Test Krip Loss')
    plt.legend()
    if tdyad is None:
        plt.title('Dyad ' + str(dyad))
    else:
        plt.title('Validation Dyad ' + str(dyad) + '\t' + 'Test Dyad ' + str(tdyad))
    plt.savefig(img_path)
