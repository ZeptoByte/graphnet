
from graphnet.models.gnn.gnn import GNN

from torch_geometric.data import Data
from torch import Tensor

class EHETransformer(GNN):
    """EHETransformer model"""

    def __init__(
        self,
        n_features: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            n_features: The number of features in the input data.
            dropout: The dropout rate to apply to the model.
        """
        super().__init__()

        self.dropout = dropout

    def forward(self, data: Data) -> Tensor:
        
        x = data.x

        return x
