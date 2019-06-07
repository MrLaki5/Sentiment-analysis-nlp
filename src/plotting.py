import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

LABELS_TWO_CLASS = ['Negative', 'Positive']
LABELS_THREE_CLASS = ['Negative', 'Neutral', 'Positive']


def calculate_confusion_matrix(y_true, y_pred, class_labels):
    cm = confusion_matrix(y_true, y_pred)
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Greens')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_labels)
    ax.yaxis.set_ticklabels(class_labels)
    return cm


def calculate_normalized_confusion_matrix(y_true, y_pred, class_labels, title="Normalized confusion matrix"):
    cm = confusion_matrix(y_true, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = plt.subplot()
    sns.heatmap(cm, annot=True, ax=ax, cmap='Greens')
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_ticklabels(class_labels)
    ax.yaxis.set_ticklabels(class_labels)
    return cm

def show_confusion_matrix():
    plt.show()

#y_true = [1, 1, 2, 2, 1, 0]
#y_pred = [1, 1, 2, 1, 2, 0]
#calculate_normalized_confusion_matrix(y_true, y_pred, LABELS_THREE_CLASS)
#show_confusion_matrix()
#calculate_confusion_matrix(y_true2, y_pred2, LABELS_TWO_CLASS)
#show_confusion_matrix()