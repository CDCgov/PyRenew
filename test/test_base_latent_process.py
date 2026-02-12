"""
Unit tests for BaseLatentInfectionProcess.
"""

import jax.numpy as jnp
import pytest

from pyrenew.deterministic import DeterministicPMF
from pyrenew.latent.base import BaseLatentInfectionProcess


class TestPopulationStructureParsing:
    """Test _parse_and_validate_fractions static method."""

    def test_parse_subpop_fractions(self):
        """Test parsing subpop_fractions."""
        pop = BaseLatentInfectionProcess._parse_and_validate_fractions(
            subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
        )

        assert pop.n_subpops == 3
        assert jnp.allclose(pop.fractions, jnp.array([0.3, 0.25, 0.45]))

    def test_rejects_fractions_not_summing_to_one(self):
        """Test that fractions not summing to 1 raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                subpop_fractions=jnp.array([0.3, 0.25, 0.40]),
            )

    def test_rejects_negative_fractions(self):
        """Test that negative fractions raise error."""
        with pytest.raises(ValueError, match="must be non-negative"):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                subpop_fractions=jnp.array([0.3, -0.1, 0.8]),
            )

    def test_rejects_missing_fractions(self):
        """Test that missing fractions raise error."""
        with pytest.raises(ValueError, match="subpop_fractions must be provided"):
            BaseLatentInfectionProcess._parse_and_validate_fractions()

    def test_rejects_2d_fractions(self):
        """Test that 2D fraction arrays raise error."""
        with pytest.raises(ValueError, match="must be a 1D array"):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                subpop_fractions=jnp.array([[0.3, 0.25, 0.45]]),
            )

    def test_rejects_empty_subpopulations(self):
        """Test that empty subpopulations raise error."""
        with pytest.raises(ValueError, match="Must have at least one subpopulation"):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                subpop_fractions=jnp.array([]),
            )


class TestBaseLatentInfectionProcessInit:
    """Test BaseLatentInfectionProcess initialization."""

    def test_rejects_missing_gen_int_rv(self):
        """Test that None gen_int_rv is rejected."""
        with pytest.raises(ValueError, match="gen_int_rv is required"):

            class ConcreteLatent(BaseLatentInfectionProcess):
                def validate(self):
                    pass

                def sample(self, n_days_post_init, **kwargs):
                    pass

            ConcreteLatent(
                name="test_latent", gen_int_rv=None, n_initialization_points=3
            )

    def test_rejects_insufficient_n_initialization_points(self):
        """Test that n_initialization_points < gen_int length is rejected."""
        with pytest.raises(
            ValueError, match="n_initialization_points must be at least"
        ):

            class ConcreteLatent(BaseLatentInfectionProcess):
                def validate(self):
                    pass

                def sample(self, n_days_post_init, **kwargs):
                    pass

            ConcreteLatent(
                name="test_latent",
                gen_int_rv=DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3])),
                n_initialization_points=2,
            )


class TestValidateI0:
    """Test _validate_I0 method."""

    def test_validate_I0_rejects_values_greater_than_one(self):
        """Test that I0 values > 1 are rejected."""
        with pytest.raises(ValueError, match="I0 must be <= 1"):
            BaseLatentInfectionProcess._validate_I0(jnp.array(1.5))

    def test_validate_I0_rejects_zero(self):
        """Test that I0 = 0 is rejected."""
        with pytest.raises(ValueError, match="I0 must be positive"):
            BaseLatentInfectionProcess._validate_I0(jnp.array(0.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
