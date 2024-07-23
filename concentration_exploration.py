# numpydoc ignore=GL08
import arviz as az
import jax.random as random
import numpyro
import numpyro.distributions as dist
import numpyro.distributions.transforms as transforms
import polars as pl
from numpyro.infer import MCMC, NUTS

# Set up the initial conditions
dist_mean = 1000
n_data_samples = 1000
n_mcmc_samples = 1000
n_mcmc_warmup = 1000
n_mcmc_chains = 4
numpyro.set_host_device_count(n_mcmc_chains)
rng_key = random.PRNGKey(0)
data_concentration = 2000

data_samples = dist.NegativeBinomial2(
    mean=dist_mean, concentration=data_concentration
).sample(rng_key, (n_data_samples,))


def my_model(concentration_dist, transform_after_sample=False):
    # numpydoc ignore=GL08
    concentration = numpyro.sample("concentration_raw", concentration_dist)
    if transform_after_sample:
        concentration = transforms.PowerTransform(-2)(concentration)
    numpyro.deterministic("concentration", concentration)
    numpyro.sample(
        "obs",
        dist.NegativeBinomial2(mean=dist_mean, concentration=concentration),
        obs=data_samples,
    )


# Define the combinations to test
combinations = [
    {"name": "HalfNormal", "dist": dist.HalfNormal(), "transform": True},
    {
        "name": "TransformedHalfNormal",
        "dist": dist.TransformedDistribution(
            dist.HalfNormal(), transforms.PowerTransform(-2)
        ),
        "transform": False,
    },
    {
        "name": "TruncatedNormal",
        "dist": dist.TruncatedNormal(low=0.1),
        "transform": True,
    },
    {
        "name": "TruncatedTransformedNormal",
        "dist": dist.TransformedDistribution(
            dist.TruncatedNormal(low=0.1),
            transforms.PowerTransform(-2),
        ),
        "transform": False,
    },
]

# Run MCMC for each combination and store results
results = []
for combo in combinations:
    nuts_kernel = NUTS(my_model)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=n_mcmc_warmup,
        num_samples=n_mcmc_samples,
        num_chains=n_mcmc_chains,
    )
    mcmc.run(
        rng_key,
        concentration_dist=combo["dist"],
        transform_after_sample=combo["transform"],
    )

    # Extract posterior samples
    samples = mcmc.get_samples()

    # Calculate summary statistics
    summary = az.summary(
        az.from_numpyro(mcmc), var_names=["concentration"], stat_focus="median"
    )

    results.append(
        {
            "name": combo["name"],
            "transform": combo["transform"],
            **summary.loc["concentration"],
        }
    )

# Create a Polars DataFrame from the results
df = pl.DataFrame(results)

# Display the results
print(df)

# You can now save the DataFrame to a CSV file if needed
df.write_csv("mcmc_results.csv")
