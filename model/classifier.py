import torch
from torch import nn
from model.TSD_model.TSD import TSD_B, TSD_T
from model.TSD_model.TSD_blocks import Mlp

from utils.utils import count_parameters


class TSD_Classifier(nn.Module):
    """
    Classification head built on top of the TSD-Tiny (TSD_T) backbone.

    Applies global average pooling to the sequence of token embeddings and maps to
    class probabilities via a small MLP classifier.

    Args:
        num_classes (int): Number of output classes.
        dim (int): Embedding dimension of the backbone output. Defaults to 128.

    Inputs:
        x (Tensor): Input image tensor of shape (B, C, H, W).
        return_logits (bool): If True, returns raw class logits instead of softmax probabilities.

    Returns:
        Tensor:
            - If return_logits is False: class probabilities of shape (B, num_classes)
            - If return_logits is True: raw logits of shape (B, num_classes)
    """

    def __init__(
        self,
        num_classes,
        dim=128,
        num_heads=4,
        num_encoder_layers=4,
        expansion_ratio=2,
        cte_out_channels=64,
    ):
        super().__init__()
        self.num_classes = num_classes
        # self.backbone = TSD_T(dim, cte_out_channels, num_heads, num_encoder_layers, expansion_ratio)

        self.backbone = TSD_B(
            dim, cte_out_channels, num_heads, num_encoder_layers, expansion_ratio)
        self.pool = nn.AdaptiveAvgPool1d(1)
        if num_classes > 2:
            self.classifier = Mlp(
                in_features=dim, hidden_features=512, out_features=num_classes
            )
        else:
            self.classifier = Mlp(in_features=dim, hidden_features=512, out_features=1)

    def forward(self, x, return_logits=False):
        x = self.backbone(x)
        x = x.permute(0, 2, 1)
        x = self.pool(x).squeeze(-1)
        logits = self.classifier(x)
        if return_logits:
            return logits
        return torch.softmax(logits, dim=1)


if __name__ == "__main__":
    model = TSD_Classifier(10)
    print(count_parameters(model=model))
