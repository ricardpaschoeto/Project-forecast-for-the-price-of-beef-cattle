"""
Market module initializer.

Expõe os principais componentes do submódulo market:
- APIs de mercado (Nasdaq, Alpha Vantage)
- APIs de comércio exterior (Comex, WITS)
- Função de montagem de tabela de mercado (market_builder)
"""

from .nasdaq_api import (
    fetch_nasdaq_timeseries,
    fetch_nasdaq_first_available,
)

from .alpha_api import (
    fetch_alpha_fx_daily,
    fetch_alpha_commodity,
)

from .comex_api import (
    comex_get_block_country_codes,
    comex_query_exports_monthly,
)

from .wits_api import (
    wits_country_map,
    wits_code_by_iso3,
    wits_fetch_tariffs_trn,
    tariff_effective_min_mfn_pref,
)

from .market_builder import (
    get_market_data,
    fill_beef_trade_and_tariffs,
)