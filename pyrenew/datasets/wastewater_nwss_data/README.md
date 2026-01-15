# Wastewater NWSS Data

## Provenance

- **Source Repository**: [CDC cfa-forecast-renewal-ww](https://github.com/cdcgov/cfa-forecast-renewal-ww)
- **Original Location**: `cfaforecastrenewalww/inst/testdata/`
- **License**: Public Domain (CC0 1.0 Universal) - U.S. Government work
- **Date Extracted**: January 2025

## Files

### `fake_nwss.csv`

Synthetic wastewater surveillance data in NWSS (National Wastewater Surveillance System) format. Contains deliberately added noise for public release.

- **Jurisdictions**: CA, WA, NM (real states) plus XX, YY, ZZ (fictional)
- **WWTPs**: CA (5), WA (4), NM (2), others (4 each)
- **Date range**: 2023-01-01 to 2023-11-06 (~310 days)
- **Size**: 487 KB, 3,286 rows
- **Granularity**: Site-lab-date level (multiple labs per WWTP)
- **Use case**: Tutorials, multi-signal model development

### Schema

| Column | Type | Description |
|--------|------|-------------|
| `wwtp_jurisdiction` | string | State/territory abbreviation |
| `wwtp_name` | string | Wastewater treatment plant identifier |
| `county_names` | string | County code |
| `lab_id` | integer | Laboratory identifier |
| `population_served` | integer | Population served by this WWTP |
| `sample_location` | string | Sample collection point (e.g., "wwtp") |
| `sample_matrix` | string | Sample type (e.g., "raw wastewater") |
| `pcr_target_units` | string | Measurement units |
| `pcr_target` | string | Target pathogen (always "sars-cov-2") |
| `pcr_target_avg_conc` | float | Viral RNA concentration |
| `lod_sewage` | float | Limit of detection for this sample |
| `pcr_target_below_lod` | integer | Below detection limit flag (0=above, 1=below) |
| `sample_collect_date` | string | Sample collection date (YYYY-MM-DD) |
| `quality_flag` | string | Data quality flags |

## Usage

```python
from pyrenew import datasets

ca_ww = datasets.load_wastewater_data_for_state("CA")
```

## Notes

- Some measurements are in linear scale, others in log10 scale (check `pcr_target_units`)
- The loader function handles unit standardization and below-LOD substitution
