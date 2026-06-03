import numpy as np
import random
import click
import torch
import time
from typing import Dict, Tuple, Optional
from pathlib import Path

from datasets.dataset import get_dataloader
from utils import (
    EarlyStopping,
    AverageMeter,
    print_metrics,
    log_metrics_df,
    plot_learning_curves_from_df,
    plot_confusion_matrix,
    save_metrics,
    get_metrics_clf,
    compute_class_weights,
)
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
)
from models.build_model import build_network
import warnings
import logging
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")

# Existing logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_file_logging(directory):
    log_file = directory / "experiment.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    class_weights: Optional[torch.Tensor],
    scaler: Optional[GradScaler] = None,
    use_amp: bool = False,
) -> Tuple[Dict[str, float], float]:
    """One epoch trainer for the hyperspectral shadow removal model"""
    model.train()
    metrics = {
        "loss": AverageMeter(),
        "f1": AverageMeter(),
        "acc": AverageMeter(),
        "precision": AverageMeter(),
        "recall": AverageMeter(),
        "balanced acc": AverageMeter(),
    }

    start_time = time.time()
    for data, labels in dataloader:
        data = data.float().to(device)
        labels = labels.long().to(device)

        data, labels = model.preprocess(data, labels)

        optimizer.zero_grad()

        with autocast(enabled=use_amp):
            loss, logits = model.get_loss(data, labels, class_weights, return_logits=True)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits.cpu().detach(), dim=-1).flatten()
        labels = labels.cpu().flatten()

        batch_size = data.shape[0]
        metrics["loss"].update(loss.item(), batch_size)
        metrics["f1"].update(f1_score(labels, preds, average="macro"), batch_size)
        metrics["acc"].update(accuracy_score(labels, preds), batch_size)
        metrics["precision"].update(
            precision_score(labels, preds, average="macro", zero_division=0), batch_size
        )
        metrics["recall"].update(
            recall_score(labels, preds, average="macro", zero_division=0), batch_size
        )
        metrics["balanced acc"].update(balanced_accuracy_score(labels, preds), batch_size)

    return {
        k: v.avg * (100 if k != "loss" else 1) for k, v in metrics.items()
    }, time.time() - start_time


def validate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[Dict[str, float], float]:
    """Full set validation for the hyperspectral shadow removal model with optional mixed precision."""
    model.eval()
    metrics = {
        "loss": AverageMeter(),
        "f1": AverageMeter(),
        "acc": AverageMeter(),
        "precision": AverageMeter(),
        "recall": AverageMeter(),
        "balanced acc": AverageMeter(),
    }

    start_time = time.time()
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.float().to(device)
            labels = labels.long().to(device)

            data, labels = model.preprocess(data, labels)

            with autocast(enabled=use_amp):
                loss, logits = model.get_loss(data, labels, return_logits=True)

            preds = torch.argmax(logits.cpu().detach(), dim=-1).flatten()
            labels = labels.cpu().flatten()

            batch_size = data.shape[0]
            metrics["loss"].update(loss.item(), batch_size)
            metrics["f1"].update(f1_score(labels, preds, average="macro"), batch_size)
            metrics["acc"].update(accuracy_score(labels, preds), batch_size)
            metrics["precision"].update(
                precision_score(labels, preds, average="macro", zero_division=0), batch_size
            )
            metrics["recall"].update(
                recall_score(labels, preds, average="macro", zero_division=0), batch_size
            )
            metrics["balanced acc"].update(balanced_accuracy_score(labels, preds), batch_size)

    return {
        k: v.avg * (100 if k != "loss" else 1) for k, v in metrics.items()
    }, time.time() - start_time


def get_flat_preds(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Obtaining flat predictions and labels from the model with optional mixed precision."""
    preds, labels = [], []

    model.eval()
    with torch.no_grad():
        for data, label in dataloader:
            data = data.float().to(device)
            label = label.long().to(device)

            data, label = model.preprocess(data, label)

            with autocast(enabled=use_amp):
                pred = model.predict(data)

            pred = pred.detach().cpu().flatten()
            label = label.cpu().flatten()

            preds.append(pred)
            labels.append(label)

    return torch.cat(preds).numpy().astype(int), torch.cat(labels).numpy().astype(int)


def prediction_model_with_patches(
    model_name, model, dataloader, patch_size=224, stride=112, device="cuda", num_classes=3
):
    """
    Evaluates a computer vision model on images of varying sizes by splitting them into patches.

    Args:
        model (nn.Module): The PyTorch model to evaluate
        dataloader: PyTorch dataloader with batch_size=1 containing images of varying sizes
        patch_size (int): Size of the patches to extract (default: 224)
        stride (int): Stride between patches (default: 112, which gives 50% overlap)
        device (str): Device to run the model on ('cuda' or 'cpu')
        num_classes (int): Number of classes for prediction

    Returns:
        tuple: Lists of predictions, images, and ground truth masks
    """
    model.eval()
    model = model.to(device)
    preds = []
    masks = []

    with torch.no_grad():
        for batch in dataloader:
            # Assuming batch contains images of shape [1, C, H, W] and masks
            image = batch[0][0].float().to(device)  # Remove batch dimension
            mask = batch[1][0].long()  # Remove batch dimension

            # Get image dimensions (should be C, H, W for PyTorch tensors)
            h, w, c = image.shape

            prediction_map = torch.zeros((num_classes, h, w), device=device)
            count_map = torch.zeros((1, h, w), device=device)

            # Extract and process patches with edge coverage
            # Calculate the y positions including the last patch that might overlap with the edge
            y_positions = list(range(0, h - patch_size + 1, stride))
            if h > patch_size and (h - patch_size) % stride != 0:
                y_positions.append(h - patch_size)  # Add the last valid y position

            # Calculate the x positions including the last patch that might overlap with the edge
            x_positions = list(range(0, w - patch_size + 1, stride))
            if w > patch_size and (w - patch_size) % stride != 0:
                x_positions.append(w - patch_size)  # Add the last valid x position

            for y in y_positions:
                for x in x_positions:
                    # Extract patch
                    patch = image[y : y + patch_size, x : x + patch_size, :].unsqueeze(0).to(device)
                    patch, _ = model.preprocess(patch, mask)
                    # Get model prediction
                    patch_pred = model(patch)

                    if model_name != "unetv1":
                        patch_pred = patch_pred.permute(0, 3, 1, 2)

                    prediction_map[:, y : y + patch_size, x : x + patch_size] += patch_pred.squeeze(
                        0
                    )
                    count_map[0, y : y + patch_size, x : x + patch_size] += 1

            # Average the predictions
            # Add small epsilon to avoid division by zero
            eps = 1e-10
            avg_prediction = prediction_map / (count_map + eps)

            # For a tensor of shape [num_classes, h, w]:
            # 1. Apply softmax along dim=0 (class dimension) to get probabilities for each pixel
            # This normalizes across classes so that probabilities sum to 1.0 at each spatial location
            softmax_prediction = torch.softmax(avg_prediction, dim=0)

            # 2. Use argmax along dim=0 to find the class index with highest probability at each pixel
            #    This produces a tensor of shape [h, w] containing class indices
            final_prediction = torch.argmax(softmax_prediction, dim=0).cpu().numpy()

            preds.append(final_prediction)
            masks.append(mask.cpu().numpy())
    return preds, masks


def save_checkpoint(model, optimizer, epoch, metrics, directory):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
    }
    torch.save(checkpoint, Path(directory) / "checkpoint_best.pth")


@click.command()
# Data parameters
@click.option("--data_dir", default="../data/cloud_shadows", help="Training data directory")
@click.option("--run_dir", default="../experiments", help="Directory to save run")
@click.option("--model_name", default="logreg", help="Deep Learning model")
@click.option(
    "--fold", default=0, help="'k' fold of a 5-fold model selection strategy. Must be between [0,4]"
)
@click.option(
    "--in_dim", default=1024, help="input dimension. Default: 1024 (CH4 spectral dimension size)"
)
# Training parameters
@click.option(
    "--num_epochs",
    default=100,
    help="Number of training epochs for the hsm algorithm. Default: 200",
)
@click.option(
    "--report_every", default=5, help="Number of epochs to report performance. Default: 5"
)
@click.option("--patience", default=20, help="patience for early stopping criterion. Default: 20")
@click.option("--batch_size", default=512, help="batch size")
@click.option("--n_workers", default=4, help="Number of cpu workers for dataloading")
@click.option("--lr", default=1e-4, help="learning rate")
@click.option("--hidden_dims", default="20,20", help="mlp dims, comma separated, none for no mlp")
@click.option("--lambda_l2", default=1e-2, help="l2 regularization")
@click.option("--norm_type", default="minmax", help="norm on spectrum. Choices: ['minmax', 'std']")
@click.option("--weighted", is_flag=True, help="if weighted loss.")
@click.option("--pretrained", is_flag=True, help="if pretrained model is available.")
@click.option("--finetune", is_flag=True, help="if pretrained model is available for finetuning.")
@click.option("--use_amp", is_flag=True, help="Enable Automatic Mixed Precision training")
@click.option(
    "--finetune_dir",
    default="../experiments/",
    help="Data path if of pretrained weights if fine-tuning.",
)
def train_cli(**kwargs):
    """Main function for the HSR algorithm training."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler = GradScaler() if kwargs["use_amp"] else None

    lr_str = f"{kwargs['lr']:.0e}".replace("e-0", "e-").replace("e+0", "e")
    job_name = f"{kwargs['model_name']}_lr{lr_str}_{kwargs['norm_type']}_w{kwargs['weighted']}_f{kwargs['fold']}"
    directory = Path(kwargs["run_dir"]) / job_name
    directory.mkdir(parents=True, exist_ok=True)
    setup_file_logging(directory)

    assert Path(kwargs["data_dir"]).exists(), "Data directory does not exist"

    if "mair_cs" in kwargs["data_dir"]:
        mask_types = ["dark_surface_mask", "cloud_shadow_mask", "cloud_mask"]
    elif "msat_cs" in kwargs["data_dir"]:
        mask_types = ["cloud_shadow_mask", "cloud_mask"]
    else:
        raise ValueError(f"Invalid data directory: {kwargs['data_dir']}")

    train_loader, val_loader, test_loader, class_map = get_dataloader(
        kwargs["data_dir"],
        kwargs["norm_type"],
        mask_types,
        batch_size=kwargs["batch_size"],
        num_workers=kwargs["n_workers"],
        fold=kwargs["fold"],
    )
    num_classes = len(class_map)
    logger.info(f"Num classes: {num_classes}")
    class_weights = compute_class_weights(train_loader, device) if kwargs["weighted"] else None
    logger.info(f"Class weights: {class_weights}")

    model = build_network(
        kwargs["model_name"], kwargs["in_dim"], num_classes, kwargs["fold"], kwargs["hidden_dims"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=kwargs["lr"], weight_decay=kwargs["lambda_l2"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=kwargs["num_epochs"])

    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Using {device} device")

    if kwargs["finetune"]:
        src_dir = kwargs["finetune_dir"]
        state_dict = torch.load(f"{src_dir}/checkpoint_best.pth", map_location=device)
        model.load_state_dict(state_dict["model"], strict=True)
        print("Training parameters restored from source. Fine-tuning!")

    if kwargs["pretrained"]:
        state_dict = torch.load(directory / "checkpoint_best.pth", map_location=device)
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        init_epoch = state_dict["epoch"] + 1
        logger.info(f"Training parameters restored at epoch {init_epoch}")
    else:
        init_epoch = 0

    es = EarlyStopping(kwargs["patience"], criterion="max")
    for epoch in range(init_epoch, kwargs["num_epochs"]):
        train_metrics, elapsed_time_t = train_epoch(
            model,
            train_loader,
            device,
            optimizer,
            class_weights,
            scaler=scaler,
            use_amp=kwargs["use_amp"],
        )
        val_metrics, elapsed_time_v = validate(model, val_loader, device, use_amp=kwargs["use_amp"])

        scheduler.step()

        if epoch % kwargs["report_every"] == 0:
            logger.info(
                f"Epoch: {epoch}/{kwargs['num_epochs']}. Training time: {elapsed_time_t:.3f}"
            )
            logger.info(f"Training Metrics...")
            logger.info(f"{print_metrics(train_metrics)}")
            logger.info(f"Validation Metrics... Inference time: {elapsed_time_v:.3f}")
            logger.info(f"{print_metrics(val_metrics)}")
            logger.info("=" * 50)

        stop, is_best = es.count(val_metrics["f1"])
        if is_best:
            save_checkpoint(model, optimizer, epoch, val_metrics, directory)
            save_metrics(directory, train_metrics, "train")
            save_metrics(directory, val_metrics, "val")
        if stop:
            logger.info("Early stopping criterion met. Stopping training.")
            break

        log_metrics_df(directory, train_metrics, epoch, elapsed_time_t, mode="train")
        log_metrics_df(directory, val_metrics, epoch, elapsed_time_v, mode="val")

    plot_learning_curves_from_df(directory)
    # Restoring last best parameters.

    state_dict = torch.load(directory / "checkpoint_best.pth", map_location=device)
    model.load_state_dict(state_dict["model"])
    # Final Validation.
    logger.info("Initializing final Evaluations:\n")

    preds, labels = get_flat_preds(model, val_loader, device, use_amp=kwargs["use_amp"])
    metrics = get_metrics_clf(labels, preds)
    logger.info(f"Final validation metrics:\n{print_metrics(metrics)}")
    save_metrics(directory, metrics, "val")
    # Final testing.

    if "msat_cs" in kwargs["data_dir"] and (
        kwargs["model_name"] == "scan"
        or kwargs["model_name"] == "unet"
        or kwargs["model_name"] == "combined_mlp"
        or kwargs["model_name"] == "combined_cnn"
        or kwargs["model_name"] == "combined_attention"
    ):
        preds, labels = prediction_model_with_patches(
            kwargs["model_name"],
            model,
            test_loader,
            patch_size=224,
            stride=112,
            num_classes=num_classes,
            device=device,
        )

        flat_preds = np.concatenate([pred.flatten() for pred in preds])
        flat_labels = np.concatenate([mask.flatten() for mask in labels])

        clf_metrics = get_metrics_clf(flat_labels, flat_preds)
        logger.info(f"Final test pixel-wise classification metrics:\n{print_metrics(clf_metrics)}")
    else:
        flat_labels, flat_preds = get_flat_preds(
            model, test_loader, device, use_amp=kwargs["use_amp"]
        )
        metrics = get_metrics_clf(flat_labels, flat_preds)
        logger.info(f"Final test pixel-wise classification metrics:\n{print_metrics(metrics)}")
    save_metrics(directory, metrics, "test")
    plot_confusion_matrix(flat_preds, flat_labels, class_map, directory)


if __name__ == "__main__":
    train_cli()
