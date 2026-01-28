import datetime as dt
import numpy as np
import pandas as pd
import holidays
from dateutil.easter import easter as holidays_easter


def _mark_single_date(df: pd.DataFrame, col: str, target_date: dt.date):
    mask = df.index.date == target_date
    df.loc[mask, col] = 1


def _mark_date_range(df: pd.DataFrame, col: str, start_date: dt.date, end_date: dt.date):
    mask = (df.index.date >= start_date) & (df.index.date <= end_date)
    df.loc[mask, col] = 1


def _second_sunday(year: int, month: int) -> dt.date:
    d = dt.date(year, month, 1)
    # Monday=0 ... Sunday=6
    days_to_first_sun = (6 - d.weekday()) % 7
    first_sunday = d + dt.timedelta(days=days_to_first_sun)
    return first_sunday + dt.timedelta(days=7)


def get_calendar_features(start: dt.date, end: dt.date) -> pd.DataFrame:
    """
    Constrói todas as features sazonais e de calendário:
    - Feriados
    - Carnaval
    - Páscoa
    - Datas comerciais
    - Férias
    - Sazonalidade de carne
    - Ciclo e aspecto
    - Flag COVID (2020–2021)
    """

    idx = pd.date_range(start=start, end=end, freq="D")
    df = pd.DataFrame(index=idx)
    df.index.name = "date"

    years = range(start.year, end.year + 1)
    br_holidays = holidays.Brazil(years=years)

    # --------- COVID FLAG ---------
    covid_start = dt.date(2020, 3, 11)
    covid_end = dt.date(2021, 12, 31)
    df["CovidPeriodFlag"] = (
        (df.index.date >= covid_start) &
        (df.index.date <= covid_end)
    ).astype(int)

    # --------- SINALIZAÇÕES ADEQUADAS ---------
    # Ciclo anual
    doy = df.index.dayofyear.values
    df["ciclo"] = doy / 365.25
    df["aspecto"] = np.sin(2 * np.pi * doy / 365.25)

    # --------- Inicializa todas as flags ---------
    flags = [
        "ano_novo_flag",
        "carnaval_core_flag", "carnaval_window_flag",
        "pascoa_flag", "pascoa_window_flag",
        "dias_maes_flag", "dias_maes_weekend_flag",
        "festas_juninas_flag",
        "santo_antonio_flag", "sao_joao_flag", "sao_pedro_flag",
        "ferias_midyear_flag", "ferias_verao_flag",
        "dias_pais_flag", "dias_pais_weekend_flag",
        "independencia_flag", "independencia_window_flag",
        "dia_criancas_flag",
        "finados_flag", "finados_weekend_flag",
        "natal_flag", "natal_window_flag",
        "fim_ano_window_flag",
    ]
    for col in flags:
        df[col] = 0

    # --------- Loop por ano ---------
    for y in years:

        # Ano novo
        ano_novo = dt.date(y, 1, 1)
        _mark_single_date(df, "ano_novo_flag", ano_novo)

        # Páscoa
        pascoa = holidays_easter(y)
        pascoa = dt.date(pascoa.year, pascoa.month, pascoa.day)
        _mark_single_date(df, "pascoa_flag", pascoa)
        _mark_date_range(df, "pascoa_window_flag", pascoa - dt.timedelta(days=3), pascoa)

        # Carnaval
        carnaval_dates = [d for d, name in br_holidays.items() if d.year == y and "Carnaval" in str(name)]
        if carnaval_dates:
            carnav_tuesday = next((d for d in carnaval_dates if d.weekday() == 1), max(carnaval_dates))
            carnav_tuesday = dt.date(carnav_tuesday.year, carnav_tuesday.month, carnav_tuesday.day)
            _mark_single_date(df, "carnaval_core_flag", carnav_tuesday)
            _mark_date_range(df, "carnaval_window_flag", carnav_tuesday - dt.timedelta(days=4), carnav_tuesday)

        # Dia das Mães → 2º domingo de maio
        dia_maes = _second_sunday(y, 5)
        _mark_single_date(df, "dias_maes_flag", dia_maes)
        _mark_date_range(df, "dias_maes_weekend_flag", dia_maes - dt.timedelta(days=2), dia_maes)

        # Dia dos Pais → 2º domingo de agosto
        dia_pais = _second_sunday(y, 8)
        _mark_single_date(df, "dias_pais_flag", dia_pais)
        _mark_date_range(df, "dias_pais_weekend_flag", dia_pais - dt.timedelta(days=2), dia_pais)

        # Festas juninas
        _mark_date_range(df, "festas_juninas_flag", dt.date(y, 6, 1), dt.date(y, 6, 30))
        _mark_single_date(df, "santo_antonio_flag", dt.date(y, 6, 13))
        _mark_single_date(df, "sao_joao_flag", dt.date(y, 6, 24))
        _mark_single_date(df, "sao_pedro_flag", dt.date(y, 6, 29))

        # Férias
        _mark_date_range(df, "ferias_midyear_flag", dt.date(y, 7, 1), dt.date(y, 7, 31))
        _mark_date_range(df, "ferias_verao_flag", dt.date(y, 1, 1), dt.date(y, 1, 31))

        # Independência
        indeps = [d for d, n in br_holidays.items() if d.year == y and "Independência" in str(n)]
        indep = indeps[0] if indeps else dt.date(y, 9, 7)
        indep = dt.date(indep.year, indep.month, indep.day)

        _mark_single_date(df, "independencia_flag", indep)
        _mark_date_range(df, "independencia_window_flag", indep - dt.timedelta(days=2), indep)

        # Dia das crianças
        _mark_single_date(df, "dia_criancas_flag", dt.date(y, 10, 12))

        # Finados
        fin = dt.date(y, 11, 2)
        _mark_single_date(df, "finados_flag", fin)
        _mark_date_range(df, "finados_weekend_flag", fin - dt.timedelta(days=2), fin)

        # Natal
        natal = dt.date(y, 12, 25)
        _mark_single_date(df, "natal_flag", natal)
        _mark_date_range(df, "natal_window_flag", dt.date(y, 12, 20), natal)

        # Fim do ano
        _mark_date_range(df, "fim_ano_window_flag", dt.date(y, 12, 26), dt.date(y, 12, 31))

    # --------- Pesos sazonais de carne ---------
    month_weights = {
        1: 1.15, 2: 1.00, 3: 1.00, 4: 0.95, 5: 1.05, 6: 1.10,
        7: 1.10, 8: 1.00, 9: 0.95, 10: 1.00, 11: 1.05, 12: 1.20,
    }
    df["sazonal_carne_weight"] = df.index.month.map(month_weights).astype(float)

    return df