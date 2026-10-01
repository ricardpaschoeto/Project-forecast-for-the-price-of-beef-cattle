"""
Climate module initializer.

Expõe as principais funções do submódulo climate:
- API INMET (precipitação e temperatura)
- API TerraClimate (scPDSI)
- API NOAA (ONI - El Niño)
- Builder de clima consolidado
"""

from .inmet_api import (
    InmetClient,
    get_openmeteo_daily,
)

from .terraclimate_api import (
    TerraClimateClient,
    get_terraclimate_scpdsi,
)

from .noaa_api import (
    NoaaClient,
    get_noaa_oni,
)

from .open_meteo_api import (
    OpenMeteoClient,
    get_precipitation,
)

from .climate_builder import (
    ClimateBuilder,
    get_climate_data,
)