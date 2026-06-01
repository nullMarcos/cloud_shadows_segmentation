import os
import json
import torch
import numpy as np
import pandas as pd

import itertools
import collections
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)


def m_std_norm(x, dim=-1):
    """Mean normalization."""
    x_norm = x - torch.mean(x, dim=dim, keepdims=True)
    x_norm = x_norm / torch.std(x_norm, dim=dim, correction=0, keepdims=True)
    return x_norm


class EarlyStopping:
    """Early stopping as the convergence criterion.

    Args:
        patience (int): the model will stop if it not do improve in a patience number of epochs.
        criterion (str): 'max' or 'min' criterions depending of the minimization objective.

    Returns:
        stop (bool): if the model must stop.
        if_best (bool): if the model performance is better than the previous models.
    """

    def __init__(
        self,
        patience: int,
        criterion: str = "max",
    ):
        self.patience = patience
        self.criterion = criterion

        assert criterion in (
            "max",
            "min",
        ), "Criterion must be one between 'max' or 'min'."
        if criterion == "max":
            self.best_metric = -np.inf
        elif criterion == "min":
            self.best_metric = np.inf
        self.counter = 0

    def count(self, metric):
        if self.criterion == "max":
            is_best = bool(metric > self.best_metric)
            self.best_metric = max(metric, self.best_metric)
        elif self.criterion == "min":
            is_best = bool(metric < self.best_metric)
            self.best_metric = min(metric, self.best_metric)
        if is_best:
            self.counter = 0
        else:
            self.counter += 1
        if self.counter > self.patience:
            stop = True
        else:
            stop = False
        return stop, is_best


def print_metrics(metrics):
    """Print the metrics for a given dictionary."""
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    return metrics


def save_metrics(directory, metrics, mode="test"):
    """save all the metrics."""
    mt_dir = os.path.join(directory, "metrics_{}.json".format(mode))
    with open(mt_dir, "w") as mt:
        json.dump(metrics, mt)


class AverageMeter(object):
    """Computes and stores the average and current value of a given metric"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def log_metrics_df(base_pth, metrics, epoch, speed=None, mode="test"):
    """save all the metrics in a DataFrame format."""
    if speed is not None:
        metrics["mean_speed"] = speed
    path = f"{base_pth}/metrics_{mode}.pkl"
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        df = pd.DataFrame(columns=list(metrics.keys()))
    metrics["epoch"] = epoch
    df = df._append(metrics, ignore_index=True)
    df.to_pickle(path)


def plot_learning_curves_from_df(base_path):
    """plot learning curves from dataframe annotated logs."""
    train = pd.read_pickle(f"{base_path}/metrics_train.pkl")
    val = pd.read_pickle(f"{base_path}/metrics_val.pkl")

    epochs = train["epoch"].values
    train = train.drop(columns=["epoch"])
    val = val.drop(columns=["epoch"])

    metrics = train.columns
    fig, axs = plt.subplots(nrows=len(metrics), ncols=1, figsize=(8, 24))
    if len(metrics) == 1: 
        axs = [axs]
    
    for i, metric in enumerate(metrics):
        axs[i].plot(epochs, train[metric], c="k", label="train", linewidth=2)
        axs[i].plot(epochs, val[metric], c="b", label="val", linewidth=2)

        axs[i].set_title(f"Learning curve {metric}", fontsize=16, pad=10, fontweight='bold')
        axs[i].set_xlabel("Epoch", fontsize=12)
        axs[i].set_ylabel(metric, fontsize=12)
        
        axs[i].grid(True, linestyle="--", alpha=0.7)
        axs[i].legend(loc="best", fontsize=11)
        axs[i].tick_params(axis='both', labelsize=10)
    fig.tight_layout()
    plt.savefig(f"{base_path}/learning_curves.png", bbox_inches='tight', dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    y_pred,
    y_true,
    class_map,
    base_path,
    normalize=True,
    title=None,
    cmap=plt.cm.Blues,
    show=False,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes_names = list(class_map.keys())
    classes = list(class_map.values())

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    if normalize:
        cm = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]) * 100
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes_names, rotation=45, fontsize=24)
    plt.yticks(tick_marks, classes_names, fontsize=24)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            "%.2f" % (cm[i, j]),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=32,
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=32)
    plt.xlabel("Predicted label", fontsize=32)
    if show:
        plt.show()
    else:
        plt.savefig(f"{base_path}/conf_m.png")


def get_metrics_clf(y_test, y_pred):
    """
    Computing the classificaion metrics.
    """
    f1 = f1_score(y_test, y_pred, average="macro")
    acc = accuracy_score(y_test, y_pred)
    p = precision_score(y_test, y_pred, average="macro")
    r = recall_score(y_test, y_pred, average="macro")
    bacc = balanced_accuracy_score(y_test, y_pred)
    metrics = {
        "f1": f1 * 100,
        "acc": acc * 100,
        "precision": p * 100,
        "recall": r * 100,
        "balanced acc": bacc * 100,
    }
    return metrics


def calculate_iou_metrics(preds, masks, class_map):
    """
    Calculates IoU metrics for segmentation predictions.

    Args:
        preds (list): List of model predictions as numpy arrays
        masks (list): List of ground truth masks as numpy arrays

    Returns:
        dict: Dictionary containing IoU metrics
    """
    # Initialize metrics
    class_labels = np.unique(np.concatenate([np.unique(m) for m in masks]))
    ious = {class_map[int(cls)]: [] for cls in class_labels}

    for pred, mask in zip(preds, masks):
        # Calculate IoU for each class
        for cls in class_labels:
            pred_mask = pred == cls
            gt_mask = mask == cls

            intersection = np.logical_and(pred_mask, gt_mask).sum()
            union = np.logical_or(pred_mask, gt_mask).sum()

            if union > 0:
                iou = intersection / union
                ious[class_map[int(cls)]].append(iou)

    # Calculate mean IoU
    mean_ious = {cls: np.mean(cls_ious) if cls_ious else 0 for cls, cls_ious in ious.items()}
    mean_iou = np.mean(list(mean_ious.values()))

    return {"class_ious": mean_ious, "mean_iou": mean_iou}


def compute_class_weights(loader, device, eps=1e-6):
    """Compute class weights for the (train) dataloader."""
    class_counts = collections.defaultdict(float)
    for _, lbl in loader:
        # count number of pixels in each class
        out, counts = torch.unique(lbl, return_counts=True)
        for o, c in zip(out, counts):
            class_counts[o.item()] += c.item()

    # Computing class weights.
    total_pixels = sum((v for k, v in class_counts.items()))
    num_classes = len(class_counts)
    class_weights = torch.tensor(
        [total_pixels / (eps + class_counts[i] * num_classes) for i in range(num_classes)],
        device=device,
    )
    return class_weights
