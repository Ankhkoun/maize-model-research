"""The classification path and the training-only D noise head."""

from torch import nn
from d_temporal_moe.models.classifier import AgriFMClassifier
from d_temporal_moe.models.noise import Noise
from d_temporal_moe.runtime import AGRIFM_SOURCE, PRETRAIN


class ClassifierModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.classifier = AgriFMClassifier(AGRIFM_SOURCE, PRETRAIN)
        self.classifier.requires_grad_(False)
        self.classifier.decoder.requires_grad_(True)

    def train_mode(self):
        self.eval()
        self.classifier.decoder.train()

    def forward(self, b):
        return dict(
            logits=self.classifier.decode_features(
                *self.classifier.encode_features(b["agrifm_cube"])
            )
        )


class TeacherModel(ClassifierModel):

    def __init__(self):
        super().__init__()
        self.module = Noise()

    def train_mode(self):
        super().train_mode()
        self.module.train()
