"""Fresh AgriFM + A4 K=4 model for the fixed USA Test500 protocol."""

from .model import AgriFMA4USA
from .objective import classification_loss

__all__ = ["AgriFMA4USA", "classification_loss"]
