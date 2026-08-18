"""MiniFrontier: a readable, from-scratch educational language model."""

from minifrontier.config import ModelConfig
from minifrontier.layers import RMSNorm, SwiGLU
from minifrontier.model import MiniFrontier, ModelOutput, TransformerBlock

__all__ = [
    "MiniFrontier",
    "ModelConfig",
    "ModelOutput",
    "RMSNorm",
    "SwiGLU",
    "TransformerBlock",
]
__version__ = "0.1.0"
