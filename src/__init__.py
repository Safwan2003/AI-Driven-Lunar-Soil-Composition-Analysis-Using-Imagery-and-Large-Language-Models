"""
AI-Driven Lunar Soil Composition Analysis
==========================================

A comprehensive framework for analyzing lunar soil composition using
computer vision and Large Language Models.
"""

__version__ = "0.1.0"
__author__ = "AI-Driven Lunar Soil Composition Analysis Team"

from . import data
from . import models
from . import llm_integration
from . import visualization
from . import utils

__all__ = [
    "data",
    "models",
    "llm_integration",
    "visualization",
    "utils",
]
