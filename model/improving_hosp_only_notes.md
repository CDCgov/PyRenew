notes for improving hosp_only_ww_model

- Change I0 reference point to be immediately before the observation period. We may have some idea about the proportion of the population that is infectious at the start of the modeling period, but not 50 days before the modeling period + exponenetioal growth.
- Initial exponential growth rate prior should be positive (Not Truncated Normal(0, 0.01))
- The model should not allow us to have more infections than the population size
-
