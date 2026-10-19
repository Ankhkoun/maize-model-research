"""Public architecture imports for B1; implementation bytes remain frozen.

AgriFMA4USA combines a frozen AgriFM encoder, NativeFPN decoder and auxiliary
A4 reliability experts. Feature residual, FiLM and weighted CE are disabled
by configs/b1.yaml, so reliability does not directly alter classification.
"""
from af2_usa_a4_rf_20261005.model import AgriFMA4USA
from af2_usa_a4_rf_20261005.agrifm import AgriFMClassifier, NativeFPN256
from af2_usa_a4_rf_20261005.reliability import A4WindowReliability

__all__ = ['AgriFMA4USA', 'AgriFMClassifier', 'NativeFPN256', 'A4WindowReliability']
