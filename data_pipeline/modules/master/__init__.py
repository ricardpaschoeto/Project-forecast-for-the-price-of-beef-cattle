# data-pipeline/modules/master/__init__.py

"""
Master module initializer.

Expõe a função pública de orquestração:
- build_master_dataframe
"""

from .master_builder import build_master_dataframe