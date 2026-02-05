# Hospital Admissions Data

## Provenance

- **Source Repository**: [CDCgov/wastewater-informed-covid-forecasting](https://github.com/CDCgov/wastewater-informed-covid-forecasting)
- **Original Location**: [`cfaforecastrenewalww/inst/testdata/`](https://github.com/CDCgov/wastewater-informed-covid-forecasting/blob/7c30b603f5e57f2b728f23506fe7714d576c31c9/cfaforecastrenewalww/inst/testdata/2023-11-06.csv)
- **License**: Public Domain (CC0 1.0 Universal) - U.S. Government work
- **Date Extracted**: January 2025

## Files

### `2023-11-06.csv`

Vintaged snapshot of COVID-19 hospital admissions data as it would have been available on 2023-11-06.

- **Coverage**: California (CA) only
- **Date range**: 2023-01-01 to 2023-11-06 (~310 days)
- **Size**: 12 KB, 311 rows
- **Use case**: Tutorials, single-jurisdiction model development

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `date` | string | Date in ISO 8601 format (YYYY-MM-DD) |
| `location` | string | State 2-letter abbreviation |
| `daily_hosp_admits` | integer | Daily COVID-19 hospital admissions count |
| `pop` | integer | State population |

## Usage

```python
from pyrenew import datasets

ca_data = datasets.load_hospital_data_for_state("CA")
```
