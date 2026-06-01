import os
from typing import List, Tuple

import torch
import torch.nn as nn
from models.scan import SpectralChannelAttentionNetwork
from models.unet import Unet


class CombinedModelCNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        unet_model: nn.Module,
        san_model: nn.Module,
        cnn_channels: List[int] = [64, 32, 16],
        dropout: float = 0.2,
    ):
        """
        Combines U-Net and SAN models using a CNN to merge their predictions.

        Args:
            in_dim: Number of input channels/dimensions
            num_classes: Number of output classes
            unet_model: Pretrained U-Net model
            san_model: Pretrained SAN model
            cnn_channels: List of channel dimensions for the CNN merger
            dropout: Dropout rate for the CNN
        """
        super().__init__()

        # Load pretrained models
        self.unet = unet_model
        self.san = san_model

        # Freeze the pretrained models
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.san.parameters():
            param.requires_grad = False

        # Create CNN for combining predictions
        # Input will have 2*num_classes channels (concatenated predictions)
        cnn_layers = []
        in_channels = num_classes * 2  # Concatenated predictions from both models

        # Add convolutional layers
        for out_channels in cnn_channels:
            cnn_layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels),
                    nn.Dropout2d(dropout),
                ]
            )
            in_channels = out_channels

        # Final layer to produce class predictions
        cnn_layers.append(nn.Conv2d(in_channels, num_classes, kernel_size=1))

        self.cnn = nn.Sequential(*cnn_layers)
        self.num_classes = num_classes

    def forward(self, x):
        """
        Forward pass of the combined model with CNN merger.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            output: Output tensor of shape (B, H, W, num_classes)
        """
        # Preprocess data for each model
        x_unet, _ = self.unet.preprocess(x, None)
        x_san, _ = self.san.preprocess(x, None)

        # Get predictions from both models
        with torch.no_grad():
            unet_pred = self.unet(x_unet)  # Shape: (B, C, H, W)
            san_pred = self.san(x_san)  # Shape: (B, H, W, C)

        # Convert SAN predictions to channel-first format for CNN
        san_pred = san_pred.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)

        # Concatenate along the channel dimension
        combined = torch.cat([unet_pred, san_pred], dim=1)  # Shape: (B, 2*C, H, W)

        # Pass through CNN merger
        merged = self.cnn(combined)  # Shape: (B, C, H, W)

        # Convert back to the expected output format (B, H, W, C)
        output = merged.permute(0, 2, 3, 1)

        return output

    def get_loss(self, x, y, class_weights=None, return_logits=False, reduction="mean"):
        """
        Calculate loss for the combined model.
        """
        logits = self.forward(x)
        # Reshape logits and y for loss calculation
        B, H, W, C = logits.shape
        logits_flat = logits.reshape(-1, C)
        y_flat = y.reshape(-1)

        loss = nn.functional.cross_entropy(
            logits_flat, y_flat, weight=class_weights, reduction=reduction
        )

        if return_logits:
            return loss, logits
        return loss

    def predict(self, x):
        """Get the mask prediction."""
        return torch.argmax(self(x), dim=-1)

    def preprocess(self, x, y):
        """Preprocess the data - returns data in (B, H, W, C) format."""
        return x, y


class CombinedModelMultiScaleCNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        unet_model: nn.Module,
        san_model: nn.Module,
        base_channels: int = 64,
        dropout: float = 0.2,
    ):
        """
        Combines U-Net and SAN models using a multi-scale CNN with skip connections
        to merge their predictions.

        Args:
            in_dim: Number of input channels/dimensions
            num_classes: Number of output classes
            unet_model: Pretrained U-Net model
            san_model: Pretrained SAN model
            base_channels: Base number of channels for the CNN
            dropout: Dropout rate for the CNN
        """
        super().__init__()

        # Load pretrained models
        self.unet = unet_model
        self.san = san_model
        self.num_classes = num_classes

        # Freeze the pretrained models
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.san.parameters():
            param.requires_grad = False

        # Initial convolution to process concatenated features
        self.init_conv = nn.Sequential(
            nn.Conv2d(num_classes * 2, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(base_channels),
        )

        # Downsampling path
        self.down1 = self._make_down_block(base_channels, base_channels * 2, dropout)
        self.down2 = self._make_down_block(
            base_channels * 2, base_channels * 4, dropout
        )

        # Upsampling path with skip connections
        self.up1 = self._make_up_block(base_channels * 4, base_channels * 2, dropout)
        self.up2 = self._make_up_block(base_channels * 2, base_channels, dropout)

        # Final convolution to produce class predictions
        self.final_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def _make_down_block(self, in_channels, out_channels, dropout):
        """Create a downsampling block for the multi-scale CNN."""
        return nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def _make_up_block(self, in_channels, out_channels, dropout):
        """Create an upsampling block with skip connections for the multi-scale CNN."""
        return nn.ModuleDict(
            {
                "upsample": nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=2, stride=2
                ),
                "conv": nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels),
                    nn.Dropout2d(dropout),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.BatchNorm2d(out_channels),
                ),
            }
        )

    def forward(self, x):
        """
        Forward pass of the combined model with multi-scale CNN merger.

        Args:
            x: Input tensor of shape (B, H, W, C)

        Returns:
            output: Output tensor of shape (B, H, W, num_classes)
        """
        # Preprocess data for each model
        x_unet, _ = self.unet.preprocess(x, None)
        x_san, _ = self.san.preprocess(x, None)

        # Get predictions from both models
        with torch.no_grad():
            unet_pred = self.unet(x_unet)  # Shape: (B, C, H, W)
            san_pred = self.san(x_san)  # Shape: (B, H, W, C)

        # Convert SAN predictions to channel-first format
        san_pred = san_pred.permute(0, 3, 1, 2)  # Convert to (B, C, H, W)

        # Concatenate along the channel dimension
        combined = torch.cat([unet_pred, san_pred], dim=1)  # Shape: (B, 2*C, H, W)

        # Apply multi-scale CNN with skip connections
        x1 = self.init_conv(combined)

        # Downsampling path
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Upsampling path with skip connections
        x = self.up1["upsample"](x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up1["conv"](x)

        x = self.up2["upsample"](x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2["conv"](x)

        # Final convolution
        x = self.final_conv(x)

        # Convert back to the expected output format (B, H, W, C)
        output = x.permute(0, 2, 3, 1)

        return output

    def get_loss(self, x, y, class_weights=None, return_logits=False, reduction="mean"):
        """
        Calculate loss for the combined model.
        """
        logits = self.forward(x)
        # Reshape logits and y for loss calculation
        B, H, W, C = logits.shape
        logits_flat = logits.reshape(-1, C)
        y_flat = y.reshape(-1)

        loss = nn.functional.cross_entropy(
            logits_flat, y_flat, weight=class_weights, reduction=reduction
        )

        if return_logits:
            return loss, logits
        return loss

    def predict(self, x):
        """Get the mask prediction."""
        return torch.argmax(self(x), dim=-1)

    def preprocess(self, x, y):
        """Preprocess the data - returns data in (B, H, W, C) format."""
        return x, y


class CombinedModelCrossAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        unet_model: nn.Module,
        san_model: nn.Module,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.2,
    ):
        """
        Combines U-Net and SAN models using a Cross-Attention mechanism.
        """
        super().__init__()

        self.unet = unet_model
        self.san = san_model
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Freeze pretrained models
        for param in self.unet.parameters():
            param.requires_grad = False
        for param in self.san.parameters():
            param.requires_grad = False

        # Proyecciones lineales para llevar las predicciones al tamaño de embedding
        # U-Net entrega (B, num_classes, H, W)
        self.proj_unet = nn.Conv2d(num_classes, embed_dim, kernel_size=1)
        # SAN entrega (B, H, W, num_classes)
        self.proj_san = nn.Conv2d(num_classes, embed_dim, kernel_size=1)

        # Capa de normalización y salida final
        self.norm = nn.LayerNorm(embed_dim)
        self.final_conv = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x):
        """
        Forward pass usando Cross-Attention.
        x: Input tensor de forma (B, H, W, C)
        """
        B, H, W, _ = x.shape

        # Obtener predicciones congeladas de los expertos
        x_unet, _ = self.unet.preprocess(x, None)
        x_san, _ = self.san.preprocess(x, None)

        with torch.no_grad():
            unet_pred = self.unet(x_unet)  # (B, num_classes, H, W)
            san_pred = self.san(x_san)  # (B, H, W, num_classes)

        # SAN sigue la forma Channel-First: (B, num_classes, H, W)
        san_pred = san_pred.permute(0, 3, 1, 2)

        # Proyectar al espacio de embedding común
        query_feat = self.proj_unet(unet_pred)  # Querys desde U-Net: (B, 64, H_unet, W_unet)
        kv_feat = self.proj_san(san_pred)  # Keys y Values desde SAN: (B, 64, H_san, W_san)
        
        # PARCHE:
        # Comparar las dimensiones de ambos expertos
        # Si difieren, interpolamos SCAN al tamano de U-Net
        H_target, W_target = query_feat.shape[2], query_feat.shape[3]
        if kv_feat.shape[2:] != query_feat.shape[2:]:
            kv_feat = torch.nn.functional.interpolate(
                kv_feat, 
                size=(H_target, W_target), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Actualizamos H y W espaciales para la reconstrucción final del tensor
        H, W = H_target, W_target

        # Se colapsa el espacio 2D (H, W) en una dimensión lineal (H*W).
        # Salidas tienen forma: (B, 64, H*W)
        Q = query_feat.flatten(2)  # Matriz Query (Origen: U-Net)
        K = kv_feat.flatten(2)  # Matriz Key   (Origen: SAN)
        V = kv_feat.flatten(2)  # Matriz Value (Origen: SAN)

        # Cálculo manual de cross attention
        # A. Calcular matriz de afinidad cruzada canal a canal (Q x K^T)
        # Operación: (B, 64, H*W) x (B, H*W, 64) -> Resultado: (B, 64, 64)
        attn_scores = torch.bmm(Q, K.transpose(-2, -1))
        
        # B. Escalamiento por la raíz del área espacial para estabilizar gradientes
        attn_scores = attn_scores / torch.sqrt(torch.tensor(H * W, dtype=torch.float32, device=x.device))
        
        # C. Softmax para convertir las puntuaciones de afinidad en distribuciones de probabilidad
        attn_weights = torch.functional.F.softmax(attn_scores, dim=-1) # (B, 64, 64)

        # D. Ponderar los valores de SAN usando los pesos de atención calculados
        # Operación: (B, 64, 64) x (B, 64, H*W) -> Resultado: (B, 64, H*W)
        attn_output = torch.bmm(attn_weights, V) 

        # Conexión residual: Sumar los mapas originales de U-Net (Q) para no perder su guía
        x_attn = attn_output + Q  # (B, 64, H*W)
        # Permutación temporal porque LayerNorm evalúa la última dimensión del tensor
        x_attn = x_attn.permute(0, 2, 1)  # (B, H*W, 64)
        x_attn = self.norm(x_attn)
        x_attn = x_attn.permute(0, 2, 1)  # Volvemos a (B, 64, H*W)

        # Deshacer el aplanado: Reconstruir la forma de imagen bidimensional (B, 64, H, W)
        x_spatial = x_attn.reshape(B, self.embed_dim, H, W)

        # Reducir el canal de embedding al número de clases: (B, num_classes, H, W)
        out = self.final_conv(x_spatial) 

        # PARCHE
        # Si la imagen procesada perdió píxeles por el camino, la interpolamos al tamaño
        # exacto de la entrada 'x'. Esto evita colapsos de dimensiones en scikit-learn.
        H_orig, W_orig = x.shape[1], x.shape[2]
        if out.shape[2:] != (H_orig, W_orig):
            out = torch.nn.functional.interpolate(
                out, 
                size=(H_orig, W_orig), 
                mode='bilinear', 
                align_corners=False
            )

        # Pasar a Channel-Last (Estándar final esperado por el pipeline): (B, H_orig, W_orig, num_classes)
        output = out.permute(0, 2, 3, 1)  # (B, H, W, num_classes)

        return output

    def get_loss(self, x, y, class_weights=None, return_logits=False, reduction="mean"):
        """
        Calcula la función de pérdida Entropía Cruzada sobre los píxeles aplanados.
        """
        logits = self.forward(x) # Obtener predicciones (B, H, W, num_classes)
        B, H, W, C = logits.shape

        # Aplanar para cumplir con el formato requerido por CrossEntropyLoss en PyTorch
        logits_flat = logits.reshape(-1, C) # (B * H * W, num_classes)
        y_flat = y.reshape(-1)              # (B * H * W)
        
        # Calcular pérdida aplicando los pesos de penalización por desbalance (class_weights)
        loss = nn.functional.cross_entropy(
            logits_flat, y_flat, weight=class_weights, reduction=reduction
        )
        if return_logits:
            return loss, logits
        return loss

    def predict(self, x):
        return torch.argmax(self(x), dim=-1)

    def preprocess(self, x, y):
        return x, y


def create_combined_model_cnn(
    in_dim: int,
    num_classes: int,
    fold: int,
    embed_dim: int = 64,
    model_type: str = "cnn",
):
    """
    Creates and initializes the combined model with CNN merger and pretrained weights.

    Args:
        in_dim: Number of input channels/dimensions
        num_classes: Number of output classes
        fold: Cross-validation fold number
        model_type: Type of merger to use, either "cnn" for simple CNN or
                   "multiscale" for multi-scale CNN with skip connections
    """
    unet_path = f"./unetv1_lr5e-3_none_wTrue_f{fold}/checkpoint_best.pth"
    san_path = f"./scan_lr1e-3_none_wTrue_f{fold}/checkpoint_best.pth"
    # unet_path = f"../experiments/exp_full/unetv1_lr1e-3_std_full_wTrue_f{fold}/checkpoint_best.pth"
    # san_path = f"../experiments/exp_full/improvedhlr_lr1e-2_std_full_wTrue_f{fold}/checkpoint_best.pth"
    # Initialize individual models
    unet = Unet(in_dim=in_dim, num_classes=num_classes)
    san = SpectralChannelAttentionNetwork(in_dim=in_dim, num_classes=num_classes)

    # Load pretrained weights
    unet.load_state_dict(torch.load(unet_path)["model"])
    san.load_state_dict(torch.load(san_path)["model"])

    # Set models to evaluation mode
    unet.eval()
    san.eval()

    # Create combined model with selected merger type
    if model_type == "cnn":
        combined_model = CombinedModelCNN(
            in_dim=in_dim, num_classes=num_classes, unet_model=unet, san_model=san
        )
    elif model_type == "multiscale":
        combined_model = CombinedModelMultiScaleCNN(
            in_dim=in_dim, num_classes=num_classes, unet_model=unet, san_model=san
        )
    elif model_type == "combined_attention":
        combined_model = CombinedModelCrossAttention(
            in_dim=in_dim,
            num_classes=num_classes,
            unet_model=unet,
            san_model=san,
            embed_dim=embed_dim,  # experimentar subiéndolo a 128
            num_heads=4,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return combined_model
