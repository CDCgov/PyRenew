"""Static location reference data for benchmark real-data runs."""

from __future__ import annotations

LOCATION_TABLE: dict[str, dict[str, int | str]] = {
    "US": {"name": "United States", "population": 341784857},
    "AL": {"name": "Alabama", "population": 5193088},
    "AK": {"name": "Alaska", "population": 737270},
    "AZ": {"name": "Arizona", "population": 7623818},
    "AR": {"name": "Arkansas", "population": 3114791},
    "CA": {"name": "California", "population": 39355309},
    "CO": {"name": "Colorado", "population": 6012561},
    "CT": {"name": "Connecticut", "population": 3688496},
    "DE": {"name": "Delaware", "population": 1059952},
    "DC": {"name": "District of Columbia", "population": 693645},
    "FL": {"name": "Florida", "population": 23462518},
    "GA": {"name": "Georgia", "population": 11302748},
    "HI": {"name": "Hawaii", "population": 1432820},
    "ID": {"name": "Idaho", "population": 2029733},
    "IL": {"name": "Illinois", "population": 12719141},
    "IN": {"name": "Indiana", "population": 6973333},
    "IA": {"name": "Iowa", "population": 3238387},
    "KS": {"name": "Kansas", "population": 2977220},
    "KY": {"name": "Kentucky", "population": 4606864},
    "LA": {"name": "Louisiana", "population": 4618189},
    "ME": {"name": "Maine", "population": 1414874},
    "MD": {"name": "Maryland", "population": 6265347},
    "MA": {"name": "Massachusetts", "population": 7154084},
    "MI": {"name": "Michigan", "population": 10127884},
    "MN": {"name": "Minnesota", "population": 5830405},
    "MS": {"name": "Mississippi", "population": 2954160},
    "MO": {"name": "Missouri", "population": 6270541},
    "MT": {"name": "Montana", "population": 1144694},
    "NE": {"name": "Nebraska", "population": 2018006},
    "NV": {"name": "Nevada", "population": 3282188},
    "NH": {"name": "New Hampshire", "population": 1415342},
    "NJ": {"name": "New Jersey", "population": 9548215},
    "NM": {"name": "New Mexico", "population": 2125498},
    "NY": {"name": "New York", "population": 20002427},
    "NC": {"name": "North Carolina", "population": 11197968},
    "ND": {"name": "North Dakota", "population": 799358},
    "OH": {"name": "Ohio", "population": 11900510},
    "OK": {"name": "Oklahoma", "population": 4123288},
    "OR": {"name": "Oregon", "population": 4273586},
    "PA": {"name": "Pennsylvania", "population": 13059432},
    "RI": {"name": "Rhode Island", "population": 1114521},
    "SC": {"name": "South Carolina", "population": 5570274},
    "SD": {"name": "South Dakota", "population": 935094},
    "TN": {"name": "Tennessee", "population": 7315076},
    "TX": {"name": "Texas", "population": 31709821},
    "UT": {"name": "Utah", "population": 3538904},
    "VT": {"name": "Vermont", "population": 644663},
    "VA": {"name": "Virginia", "population": 8880107},
    "WA": {"name": "Washington", "population": 8001020},
    "WV": {"name": "West Virginia", "population": 1766147},
    "WI": {"name": "Wisconsin", "population": 5972787},
    "WY": {"name": "Wyoming", "population": 588753},
    "PR": {"name": "Puerto Rico", "population": 3184835},
}

LOCATION_POPULATIONS: dict[str, int] = {
    abbr: int(row["population"]) for abbr, row in LOCATION_TABLE.items()
}


def population_for_location(loc_abbr: str) -> int:  # numpydoc ignore=RT01
    """Return static population for a US location abbreviation."""
    try:
        return LOCATION_POPULATIONS[loc_abbr]
    except KeyError as exc:
        raise ValueError(
            f"No static population for {loc_abbr!r}. "
            f"Available locations: {sorted(LOCATION_POPULATIONS)}"
        ) from exc


def name_for_location(loc_abbr: str) -> str:  # numpydoc ignore=RT01
    """Return static display name for a US location abbreviation."""
    try:
        return str(LOCATION_TABLE[loc_abbr]["name"])
    except KeyError as exc:
        raise ValueError(
            f"No static name for {loc_abbr!r}. "
            f"Available locations: {sorted(LOCATION_TABLE)}"
        ) from exc
