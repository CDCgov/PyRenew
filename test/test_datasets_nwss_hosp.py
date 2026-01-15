"""
Unit tests for wastewater NWSS and hospital admissions dataset loaders.

Tests cover the load_wastewater_data_for_state and load_hospital_data_for_state
functions with various parameters and edge cases.
"""

import jax.numpy as jnp
import polars as pl
import pytest

from pyrenew.datasets import (
    load_hospital_data_for_state,
    load_wastewater_data_for_state,
)


class TestLoadWastewaterDataForState:
    """Test load_wastewater_data_for_state function."""

    def test_load_default_state_ca(self):
        """Test loading wastewater data for default state (CA)."""
        data = load_wastewater_data_for_state()

        assert "observed_conc" in data
        assert "observed_conc_linear" in data
        assert "site_ids" in data
        assert "time_indices" in data
        assert "wwtp_names" in data
        assert "dates" in data
        assert "n_sites" in data
        assert "n_obs" in data
        assert "raw_df" in data

        assert isinstance(data["observed_conc"], jnp.ndarray)
        assert isinstance(data["observed_conc_linear"], jnp.ndarray)
        assert isinstance(data["site_ids"], jnp.ndarray)
        assert isinstance(data["time_indices"], jnp.ndarray)
        assert isinstance(data["wwtp_names"], list)
        assert isinstance(data["dates"], list)
        assert isinstance(data["n_sites"], int)
        assert isinstance(data["n_obs"], int)
        assert isinstance(data["raw_df"], pl.DataFrame)

        assert data["n_obs"] > 0
        assert data["n_sites"] > 0
        assert len(data["observed_conc"]) == data["n_obs"]
        assert len(data["site_ids"]) == data["n_obs"]

    def test_load_state_wa(self):
        """Test loading wastewater data for WA state."""
        data = load_wastewater_data_for_state(state_abbr="WA")

        assert data["n_obs"] > 0
        assert data["n_sites"] > 0
        assert len(data["wwtp_names"]) == data["n_sites"]

    def test_load_state_nm(self):
        """Test loading wastewater data for NM state (has log10 units)."""
        data = load_wastewater_data_for_state(state_abbr="NM")

        assert data["n_obs"] > 0
        # NM has log10 units, so standardize_log_units should convert them
        assert jnp.all(data["observed_conc_linear"] > 0)

    def test_invalid_state_raises_error(self):
        """Test that invalid state raises ValueError."""
        with pytest.raises(ValueError, match="No wastewater data found"):
            load_wastewater_data_for_state(state_abbr="XX")

    def test_substitute_below_lod_true(self):
        """Test below-LOD substitution when enabled."""
        data = load_wastewater_data_for_state(
            state_abbr="CA",
            substitute_below_lod=True,
        )

        # All linear concentrations should be positive
        assert jnp.all(data["observed_conc_linear"] > 0)

    def test_substitute_below_lod_false(self):
        """Test behavior when below-LOD substitution is disabled."""
        data = load_wastewater_data_for_state(
            state_abbr="CA",
            substitute_below_lod=False,
        )

        # Should still have valid data
        assert data["n_obs"] > 0

    def test_standardize_log_units_false(self):
        """Test behavior when log unit standardization is disabled."""
        data = load_wastewater_data_for_state(
            state_abbr="CA",
            standardize_log_units=False,
        )

        assert data["n_obs"] > 0
        assert jnp.all(data["observed_conc_linear"] >= 0)

    def test_standardize_log_units_true_with_log10_data(self):
        """Test log unit standardization with data that has log10 units."""
        # NM has log10 units in the test data
        data = load_wastewater_data_for_state(
            state_abbr="NM",
            standardize_log_units=True,
        )

        assert data["n_obs"] > 0
        # Linear concentrations should be converted from log10
        assert jnp.all(data["observed_conc_linear"] > 0)

    def test_site_ids_are_valid_indices(self):
        """Test that site_ids are valid indices into wwtp_names."""
        data = load_wastewater_data_for_state()

        assert jnp.all(data["site_ids"] >= 0)
        assert jnp.all(data["site_ids"] < data["n_sites"])

    def test_time_indices_are_non_negative(self):
        """Test that time indices are non-negative."""
        data = load_wastewater_data_for_state()

        assert jnp.all(data["time_indices"] >= 0)

    def test_observed_conc_is_log_transformed(self):
        """Test that observed_conc is log of observed_conc_linear."""
        data = load_wastewater_data_for_state()

        # observed_conc should be approximately log(observed_conc_linear + epsilon)
        # There's an epsilon of 1e-8 added before log transform
        expected_log = jnp.log(data["observed_conc_linear"] + 1e-8)
        assert jnp.allclose(data["observed_conc"], expected_log, rtol=1e-5)


class TestLoadHospitalDataForState:
    """Test load_hospital_data_for_state function."""

    def test_load_default_state_ca(self):
        """Test loading hospital data for default state (CA)."""
        data = load_hospital_data_for_state()

        assert "daily_admits" in data
        assert "population" in data
        assert "dates" in data
        assert "n_days" in data

        assert isinstance(data["daily_admits"], jnp.ndarray)
        assert isinstance(data["population"], int)
        assert isinstance(data["dates"], list)
        assert isinstance(data["n_days"], int)

        assert data["n_days"] > 0
        assert data["population"] > 0
        assert len(data["daily_admits"]) == data["n_days"]
        assert len(data["dates"]) == data["n_days"]

    def test_daily_admits_are_non_negative(self):
        """Test that daily admissions are non-negative."""
        data = load_hospital_data_for_state()

        assert jnp.all(data["daily_admits"] >= 0)

    def test_population_is_reasonable(self):
        """Test that population is a reasonable value."""
        data = load_hospital_data_for_state()

        # California population should be in the tens of millions
        assert data["population"] > 1_000_000
        assert data["population"] < 100_000_000

    def test_dates_are_sorted(self):
        """Test that dates are sorted chronologically."""
        data = load_hospital_data_for_state()

        dates = data["dates"]
        for i in range(1, len(dates)):
            assert dates[i] > dates[i - 1]

    def test_n_days_matches_array_length(self):
        """Test that n_days matches the length of daily_admits array."""
        data = load_hospital_data_for_state()

        assert data["n_days"] == len(data["daily_admits"])
        assert data["n_days"] == len(data["dates"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
