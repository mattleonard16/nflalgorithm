"""
Models package for NFL Algorithm professional system.
"""

from .position_specific.base_model import BasePositionModel
from .position_specific.rb_model import RBModel

__all__ = ['BasePositionModel', 'RBModel'] 