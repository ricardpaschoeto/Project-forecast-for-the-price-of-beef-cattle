"""
Climate module initializer.

Expõe as principais funções do submódulo climate:
- API INMET (precipitação e temperatura)
- API TerraClimate (scPDSI)
- API NOAA (ONI - El Niño)
- Builder de clima consolidado
"""

from .inmet_api import (
    get_inmet_daily_station,
    get_inmet_precip_temp,
)

from .terraclimate_api import (
    get_terraclimate_scpdsi,
)

from .noaa_api import (
    get_noaa_oni,
)

from .climate_builder import (
    get_climate_data,
)