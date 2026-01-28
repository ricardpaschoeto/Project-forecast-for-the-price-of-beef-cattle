"""
Macro module initializer.

Expõe os principais componentes do módulo macro:
- Acesso às séries do Banco Central (SGS)
- Builder macroeconômico consolidado (selic, ipca, pib*)
"""

from .bcb_api import (
    get_bcb_series_raw,
)

from .macro_builder import (
    get_macro_data,
)