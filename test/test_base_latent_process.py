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

    def test_rejects_2d_fractions(self):
        """Test that 2D fraction arrays raise error."""
        with pytest.raises(ValueError, match="Fractions must be 1D arrays"):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                obs_fractions=jnp.array([[0.3, 0.25]]),  # 2D array
                unobs_fractions=jnp.array([0.45]),
            )

    def test_rejects_no_observed_subpopulations(self):
        """Test that empty observed subpopulations raise error."""
        with pytest.raises(
            ValueError, match="Must have at least one observed subpopulation"
        ):
            BaseLatentInfectionProcess._parse_and_validate_fractions(
                obs_fractions=jnp.array([]),  # No observed subpops
                unobs_fractions=jnp.array([1.0]),
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
            obs_fractions=jnp.array([0.5, 0.5]),
            unobs_fractions=jnp.array([]),
            K=2,
            K_obs=2,
            K_unobs=0,
        )

        # Create arrays with wrong shapes
        infections_aggregate = jnp.ones(10)  # Correct: (10,)
        infections_all = jnp.ones((10, 2))  # Correct: (10, 2)
        infections_observed = jnp.ones((10, 3))  # WRONG: should be (10, 2)
        infections_unobserved = jnp.ones((10, 0))  # Correct: (10, 0)

        with pytest.raises(ValueError, match="has incorrect shape"):
            BaseLatentInfectionProcess._validate_output_shapes(
                infections_aggregate,
                infections_all,
                infections_observed,
                infections_unobserved,
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
            observed=jnp.ones((10, 1)),
            unobserved=jnp.ones((10, 1)),
        )

        agg, all_s, obs, unobs = sample
        assert agg.shape == (10,)
        assert all_s.shape == (10, 2)
        assert obs.shape == (10, 1)
        assert unobs.shape == (10, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
