"""
Unit tests for BaseLatentInfectionProcess.
"""

import jax.numpy as jnp
import pytest

from pyrenew.latent.base import BaseLatentInfectionProcess


class TestPopulationStructureParsing:
    """Test _parse_and_validate_fractions static method."""

    def test_parse_obs_unobs_fractions(self):
        """Test parsing obs_fractions + unobs_fractions."""
        pop = BaseLatentInfectionProcess._parse_and_validate_fractions(
            obs_fractions=jnp.array([0.3, 0.25]),
            unobs_fractions=jnp.array([0.45]),
        )

        assert pop.K == 3
        assert pop.K_obs == 2
        assert pop.K_unobs == 1

    def test_rejects_fractions_not_summing_to_one(self):
        """Test that fractions not summing to 1 raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                obs_fractions=jnp.array([0.3, 0.25]),
                unobs_fractions=jnp.array([0.40]),
            )

    def test_rejects_negative_fractions(self):
        """Test that negative fractions raise error."""
        with pytest.raises(ValueError, match="must be non-negative"):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                obs_fractions=jnp.array([0.3, -0.1]),
                unobs_fractions=jnp.array([0.8]),
            )

    def test_rejects_missing_fractions(self):
        """Test that missing fractions raise error."""
        with pytest.raises(
            ValueError, match="Both obs_fractions and unobs_fractions must be provided"
        ):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                obs_fractions=jnp.array([0.3, 0.25]),
            )

        with pytest.raises(
            ValueError, match="Both obs_fractions and unobs_fractions must be provided"
        ):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                unobs_fractions=jnp.array([0.45]),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
