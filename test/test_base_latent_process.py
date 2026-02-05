"""
Unit tests for BaseLatentInfectionProcess.
"""

import jax.numpy as jnp
import pytest

from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.latent import HierarchicalInfections, RandomWalk
from pyrenew.latent.base import BaseLatentInfectionProcess, LatentSample


class TestPopulationStructureParsing:
    """Test _parse_and_validate_fractions static method."""

    def test_parse_subpop_fractions(self):
        """Test parsing subpop_fractions."""
        pop = BaseLatentInfectionProcess._parse_and_validate_fractions(
            subpop_fractions=jnp.array([0.3, 0.25, 0.45]),
        )

        assert pop.K == 3
        assert jnp.allclose(pop.fractions, jnp.array([0.3, 0.25, 0.45]))

    def test_rejects_fractions_not_summing_to_one(self):
        """Test that fractions not summing to 1 raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                subpop_fractions=jnp.array([0.3, 0.25, 0.40]),  # Sum is 0.95
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
                subpop_fractions=jnp.array([[0.3, 0.25, 0.45]]),  # 2D array
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
            # Create a minimal concrete subclass for testing
            class ConcreteLatent(BaseLatentInfectionProcess):
                def validate(self):
                    pass

                def sample(self, n_days_post_init, **kwargs):
                    pass

            ConcreteLatent(gen_int_rv=None)

    def test_rejects_negative_n_initialization_points(self):
        """Test that negative n_initialization_points is rejected."""
        with pytest.raises(
            ValueError, match="n_initialization_points must be non-negative"
        ):

            class ConcreteLatent(BaseLatentInfectionProcess):
                def validate(self):
                    pass

                def sample(self, n_days_post_init, **kwargs):
                    pass

            ConcreteLatent(
                gen_int_rv=DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3])),
                n_initialization_points=-1,
            )


class TestGetRequiredLookback:
    """Test get_required_lookback method."""

    def test_get_required_lookback_returns_gen_int_length(self):
        """Test that get_required_lookback returns generation interval length."""
        gen_int = DeterministicPMF("gen_int", jnp.array([0.2, 0.5, 0.3]))
        process = HierarchicalInfections(
            gen_int_rv=gen_int,
            I0_rv=DeterministicVariable("I0", 0.001),
            initial_log_rt_rv=DeterministicVariable("initial_log_rt", 0.0),
            baseline_temporal=RandomWalk(),
            subpop_temporal=RandomWalk(),
        )

        assert process.get_required_lookback() == 3


class TestValidateOutputShapes:
    """Test _validate_output_shapes method."""

    def test_validate_output_shapes_raises_on_mismatch(self):
        """Test that shape validation raises on incorrect shapes."""
        from pyrenew.latent.base import PopulationStructure

        pop = PopulationStructure(
            fractions=jnp.array([0.5, 0.5]),
            K=2,
        )

        # Create arrays with wrong shapes
        infections_aggregate = jnp.ones(10)  # Correct: (10,)
        infections_all = jnp.ones((10, 3))  # WRONG: should be (10, 2)

        with pytest.raises(ValueError, match="has incorrect shape"):
            BaseLatentInfectionProcess._validate_output_shapes(
                infections_aggregate,
                infections_all,
                n_total_days=10,
                pop=pop,
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

    def test_validate_I0_accepts_valid_values(self):
        """Test that valid I0 values are accepted."""
        # Should not raise
        BaseLatentInfectionProcess._validate_I0(jnp.array(0.001))
        BaseLatentInfectionProcess._validate_I0(jnp.array(1.0))
        BaseLatentInfectionProcess._validate_I0(jnp.array([0.001, 0.002]))


class TestLatentSample:
    """Test LatentSample named tuple."""

    def test_latent_sample_unpacking(self):
        """Test that LatentSample can be unpacked correctly."""
        sample = LatentSample(
            aggregate=jnp.ones(10),
            all_subpops=jnp.ones((10, 2)),
        )

        agg, all_s = sample
        assert agg.shape == (10,)
        assert all_s.shape == (10, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
