"""Human-readable feature names.

Deliberately dependency-free. This map used to live in ``src/explain.py``,
which imports lightgbm, shap, matplotlib and seaborn at module level — so the
Streamlit app was pulling the entire analysis stack just to turn
``wx_precip_6h`` into "precipitation, previous 6 h". Splitting it out keeps the
app's import chain to what it genuinely needs.
"""
from __future__ import annotations

PRETTY = {
    "te_route": "route historical late rate",
    "te_carrier_dest": "carrier x destination historical rate",
    "te_origin_sched_dep_hour": "origin x hour historical rate",
    "te_dest_sched_dep_hour": "destination x hour historical rate",
    "te_carrier_sched_dep_hour": "carrier x hour historical rate",
    "te_tailnum": "airframe historical late rate",
    "te_carrier": "carrier historical late rate",
    "te_dest": "destination historical late rate",
    "sched_dep_hour": "scheduled departure hour",
    "dep_hour_sin": "departure hour (cyclical, sin)",
    "dep_hour_cos": "departure hour (cyclical, cos)",
    "rotation_slack_min": "schedule slack in the rotation (min)",
    "rotation_gap_min": "gap since previous NYC leg (min)",
    "is_tight_turn": "tight rotation flag",
    "leg_of_day": "leg number of the day",
    "tail_legs_today": "legs the airframe flies today",
    "block_slack_min": "block time vs route median (min)",
    "sched_block_min": "scheduled block time (min)",
    "implied_speed_mph": "implied schedule speed (mph)",
    "origin_hour_deps": "departures from origin this hour",
    "origin_slot15_deps": "departures from origin, 15-min slot",
    "origin_day_deps": "departures from origin today",
    "dest_hour_arrivals": "NYC arrivals into destination this hour",
    "carrier_origin_hour_deps": "carrier departures this hour",
    "carrier_share_of_slot": "carrier share of the hourly slot",
    "nyc_hour_deps": "all NYC departures this hour",
    "wx_precip": "precipitation (in)",
    "wx_precip_3h": "precipitation, previous 3 h",
    "wx_precip_6h": "precipitation, previous 6 h",
    "wx_visib": "visibility (mi)",
    "wx_visib_min_3h": "minimum visibility, previous 3 h",
    "wx_wind_speed": "wind speed (mph)",
    "wx_wind_gust": "wind gust (mph)",
    "wx_wind_gust_max_3h": "max wind gust, previous 3 h",
    "wx_wind_dir": "wind direction (deg)",
    "wx_temp": "temperature (F)",
    "wx_dewp": "dew point (F)",
    "wx_humid": "relative humidity (%)",
    "wx_pressure": "pressure (mb)",
    "wx_pressure_change_3h": "pressure change, previous 3 h",
    "wx_low_visibility": "low-visibility flag",
    "wx_freezing": "freezing flag",
    "wx_is_precipitating": "precipitation flag",
    "plane_age": "aircraft age (years)",
    "day_of_week": "day of week",
    "day_of_year": "day of year",
    "is_holiday_period": "holiday travel period",
    "distance": "distance (mi)",
    "dest_alt": "destination elevation (ft)",
    "sched_arr_hour": "scheduled arrival hour",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name.replace("_", " "))
