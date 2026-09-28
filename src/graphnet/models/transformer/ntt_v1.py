import torch
import torch.nn as nn
from typing import Optional, Tuple

from graphnet.models.gnn.gnn import GNN

from torch_geometric.data import Data
from torch import Tensor

from pytorch_lightning import LightningModule
from torch.nn.modules import TransformerEncoder, TransformerEncoderLayer

class AttentionMechanism(LightningModule):

    def __init__(
        self,
        n_features: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.dropout = dropout
        self.encoder_layer = TransformerEncoderLayer(
            d_model=512,
            nhead=2,
            dropout=dropout,
        )

        self.transformer_encoder = TransformerEncoder(
            self.encoder_layer,
            num_layers=2,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.transformer_encoder(x)


class NeutrinoTransientTransformerFilter_v1(GNN):
    """
    Fast Transformed For Transient Selection
    """
    def __init__(
        self,
        n_features: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_features = n_features
        self._dropout = dropout

    def forward(
        self,
        data: Data,
    ) -> Tensor:

        print(data)

        return data