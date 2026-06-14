import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, classes="present", ignore=None):
    """
    Multi-class Lovasz-Softmax loss
    probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
    labels: [P] Tensor, ground truth labels (between 0 and C - 1)
    classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    ignore: void class labels
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if classes == "present" and fg.sum() == 0:
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError("Sigmoid output possible only with 1 class")
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return torch.stack(losses).mean()


def lovasz_softmax(probas, labels, classes="present", per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
    probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
    labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
    classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    per_image: compute the loss per image instead of per batch
    ignore: void class labels
    """
    if per_image:
        loss = 0
        for prob, lab in zip(probas, labels):
            loss += lovasz_softmax_flat(prob.unsqueeze(0), lab.unsqueeze(0), classes, ignore)
        return loss / probas.size(0)
    else:
        return lovasz_softmax_flat(
            probas.reshape(-1, probas.size(1)), labels.reshape(-1), classes, ignore
        )


class LovaszSoftmaxLoss(nn.Module):
    def __init__(self, classes="present", per_image=False, ignore=None):
        super(LovaszSoftmaxLoss, self).__init__()
        self.classes = classes
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, probas, labels):
        return lovasz_softmax(probas, labels, self.classes, self.per_image, self.ignore)


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
            
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi


class UnetAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv1 = self.contract_block(
            in_dim, 8, 3, 5
        )  # plumes: (in_dim, 8, 3, 5); clouds: (in_dim, 8, 5, 5)
        self.conv2 = self.contract_block(8, 16, 3, 2)
        self.conv3 = self.contract_block(16, 32, 3, 1)

        self.upconv3 = self.expand_block(32, 16, 3, 1)
        
        # Attention Gate for upconv2
        # g comes from upconv3 (16 channels), x comes from conv2 (16 channels)
        self.ag2 = AttentionGate(F_g=16, F_l=16, F_int=8)
        self.upconv2 = self.expand_block(16 * 2, 8, 4, 1)
        
        # Attention Gate for upconv1
        # g comes from upconv2 (8 channels), x comes from conv1 (8 channels)
        self.ag1 = AttentionGate(F_g=8, F_l=8, F_int=4)
        self.upconv1 = self.expand_block(
            8 * 2, num_classes, 7, 1
        )  # plumes: (8 * 2, num_classes, 7, 1); clouds: (8 * 2, num_classes, 6, 1)
        
        # Initialize Lovasz loss to prevent errors when called in get_loss
        self.lovasz_loss = LovaszSoftmaxLoss()

    def forward(self, x):
        # Downsampling part
        x = self.m_std_norm(x, dim=(1, 2, 3))

        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        # Upsampling
        upconv3 = self.upconv3(conv3)
        
        # Apply attention before concatenating
        x2_att = self.ag2(g=upconv3, x=conv2)
        upconv2 = self.upconv2(torch.cat([upconv3, x2_att], 1))
        
        # Apply attention before concatenating
        x1_att = self.ag1(g=upconv2, x=conv1)
        upconv1 = self.upconv1(torch.cat([upconv2, x1_att], 1))
        
        return upconv1

    def contract_block(self, in_channels, out_channels, kernel_size, padding):

        contract = nn.Sequential(
            torch.nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):

        expand = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1
            ),
        )
        return expand

    def get_loss(
        self,
        x,
        y,
        class_weights=None,
        return_logits=False,
        reduction="mean",
        loss_type="ce",
        alpha=0.5,
    ):
        """Get the loss function for the Unet Attention Model.

        Args:
            x: Input data
            y: Target labels
            class_weights: Optional weights for cross-entropy loss
            return_logits: Whether to return logits along with the loss
            reduction: Reduction method for cross-entropy
            loss_type: Type of loss ('ce' for Cross-Entropy, 'lovasz' for Lovasz-Softmax,
                       'combined' for alpha*CE + (1-alpha)*Lovasz)
            alpha: Weight for combined loss (when loss_type='combined')
        """
        logits = self(x)
        if loss_type == "ce":
            # Original cross-entropy loss
            loss = nn.functional.cross_entropy(logits, y, weight=class_weights, reduction=reduction)
        elif loss_type == "lovasz":
            # Lovasz-Softmax loss
            probas = F.softmax(logits, dim=1)
            loss = self.lovasz_loss(probas, y)
        elif loss_type == "combined":
            # Combined loss (CE + Lovasz)
            ce_loss = nn.functional.cross_entropy(
                logits, y, weight=class_weights, reduction=reduction
            )
            probas = F.softmax(logits, dim=1)
            lovasz = self.lovasz_loss(probas, y)
            loss = alpha * ce_loss + (1 - alpha) * lovasz
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if return_logits:
            return loss, logits.permute(0, 2, 3, 1)
        return loss

    def predict(self, x):
        """Get the mask prediction for the hyperspectral data."""
        return torch.argmax(self(x), dim=1)

    def preprocess(self, x, y):
        """Preprocess the data from (B H W C) to (B C H W), where C is the spectral dimension.
        Default: 1024
        """
        # resize
        x = x.permute(0, 3, 1, 2)
        return x, y

    def m_std_norm(self, x, dim=(1, 2, 3)):
        x_std = x - torch.mean(x, dim=dim, keepdims=True)
        x_std = x_std / torch.std(x_std, dim=dim, correction=0, keepdims=True)
        return x_std
