# Pyrenew demo


This demo simulates some basic renewal process data and then fits to it
using `pyrenew`.

You’ll need to install `pyrenew` first. You’ll also need working
installations of `matplotlib`, `numpy`, `jax`, `numpyro`, and `polars`

``` python
import matplotlib as mpl
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed
import numpyro.distributions as dist
```

``` python
from pyrenew.process import SimpleRandomWalkProcess
```

    An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.

``` python
np.random.seed(3312)
q = SimpleRandomWalkProcess(dist.Normal(0, 0.001))
with seed(rng_seed=np.random.randint(0,1000)):
    q_samp = q.sample(duration=100)

plt.plot(np.exp(q_samp[0]))
```

<img
src="pyrenew_demo_files/figure-commonmark/fig-randwalk-output-1.png"
id="fig-randwalk" />

``` python
from pyrenew.latent import Infections, HospitalAdmissions
from pyrenew.observation import PoissonObservation, DeterministicObs
from pyrenew.model import HospitalizationsModel
from pyrenew.process import RtRandomWalkProcess

# Initializing model components:

# A deterministic generation time
gen_time = DeterministicObs(
    (jnp.array([0.25, 0.25, 0.25, 0.25]),),
    validate_pmf=True
    )

# The latent infections process
latent_infections = Infections(gen_int=gen_time)

# A deterministic infection to hosp pmf
inf_hosp_int = DeterministicObs(
    (jnp.array([0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0.25, 0.5, 0.1, 0.1, 0.05]),),
    validate_pmf=True
    )

# The latent hospitalization process
latent_hospitalizations = HospitalAdmissions(inf_hosp_int=inf_hosp_int)

# And observation process for the hospitalizations
observed_hospitalizations = PoissonObservation(
    rate_varname='latent',
    counts_varname='observed_hospitalizations',
    )

# And a random walk process (it could be deterministic using
# pyrenew.process.DeterministicProcess())
Rt_process = RtRandomWalkProcess()

# Initializing the model
hospmodel = HospitalizationsModel(
    latent_hospitalizations=latent_hospitalizations,
    observed_hospitalizations=observed_hospitalizations,
    latent_infections=latent_infections,
    Rt_process=Rt_process
    )
```

``` python
with seed(rng_seed=np.random.randint(1, 60)):
    x = hospmodel.sample(constants=dict(n_timepoints=30))
x
```

    HospModelSample(Rt=Array([1.1791104, 1.1995267, 1.1772177, 1.1913829, 1.2075942, 1.1444623,
           1.1514508, 1.1976782, 1.2292639, 1.1719677, 1.204649 , 1.2323451,
           1.2466507, 1.2800207, 1.2749145, 1.2619376, 1.2189837, 1.2192641,
           1.2290158, 1.2128737, 1.1908046, 1.2174997, 1.1941082, 1.2084603,
           1.1965215, 1.2248698, 1.2308019, 1.2426206, 1.2131014, 1.207159 ,
           1.1837622], dtype=float32), infections=Array([ 1.4125489,  1.8606048,  2.373585 ,  3.1091077,  2.6433773,
            2.8573434,  3.161715 ,  3.5246303,  3.74528  ,  3.893561 ,
            4.314205 ,  4.76846  ,  5.211469 ,  5.8201566,  6.4110003,
            7.0072513,  7.4510007,  8.13536  ,  8.911782 ,  9.553016 ,
           10.137069 , 11.181891 , 11.876529 , 12.9149685, 13.793039 ,
           15.239348 , 16.561634 , 18.176119 , 19.339912 , 20.919167 ,
           22.194605 ], dtype=float32), IHR=Array(0.04929917, dtype=float32), latent=Array([0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.01740937, 0.05775031,
           0.08208082, 0.11296336, 0.13357335, 0.13198984, 0.14360985,
           0.15615721, 0.16922975, 0.18031327, 0.19277987, 0.21146055,
           0.23146638, 0.25456703, 0.28231323, 0.31053045, 0.33770248,
           0.36442798], dtype=float32), sampled=Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=int32))

``` python
fig, ax = plt.subplots(nrows=3, sharex=True)
ax[0].plot(x.infections)
ax[0].set_ylim([1/5, 5])
ax[1].plot(x.latent)
ax[2].plot(x.sampled, 'o')
for axis in ax[:-1]:
    axis.set_yscale("log")
```

<img src="pyrenew_demo_files/figure-commonmark/fig-hosp-output-1.png"
id="fig-hosp" />

``` python
sim_dat={"observed_hospitalizations": x.sampled}
constants = {"n_timepoints":len(x.sampled)-1}

# from numpyro.infer import MCMC, NUTS
hospmodel.run(
    num_warmup=1000,
    num_samples=1000,
    random_variables=sim_dat,
    constants=constants,
    rng_key=jax.random.PRNGKey(54),
    mcmc_args=dict(progress_bar=False),
    )
```

``` python
hospmodel.print_summary()
```


                                     mean       std    median      5.0%     95.0%     n_eff     r_hat
                             I0      7.14      1.75      6.87      4.30      9.69   1052.68      1.00
                            IHR      0.05      0.00      0.05      0.05      0.05   1712.91      1.00
                            Rt0      1.13      0.12      1.13      0.92      1.32    960.40      1.00
     Rt_transformed_rw_diffs[0]     -0.00      0.02     -0.00     -0.04      0.04   1315.29      1.00
     Rt_transformed_rw_diffs[1]     -0.00      0.03      0.00     -0.04      0.04   1886.70      1.00
     Rt_transformed_rw_diffs[2]     -0.00      0.02     -0.00     -0.04      0.04   1460.97      1.00
     Rt_transformed_rw_diffs[3]     -0.00      0.02     -0.00     -0.04      0.04   1669.85      1.00
     Rt_transformed_rw_diffs[4]     -0.00      0.02     -0.00     -0.04      0.04   1667.25      1.00
     Rt_transformed_rw_diffs[5]      0.00      0.03      0.00     -0.05      0.04   1702.68      1.00
     Rt_transformed_rw_diffs[6]     -0.00      0.02     -0.00     -0.04      0.04   1558.87      1.00
     Rt_transformed_rw_diffs[7]     -0.00      0.02     -0.00     -0.04      0.04   2296.40      1.00
     Rt_transformed_rw_diffs[8]     -0.00      0.02     -0.00     -0.04      0.04   1850.65      1.00
     Rt_transformed_rw_diffs[9]      0.00      0.02      0.00     -0.04      0.03   1460.40      1.00
    Rt_transformed_rw_diffs[10]     -0.00      0.02      0.00     -0.04      0.04   2055.47      1.00
    Rt_transformed_rw_diffs[11]     -0.00      0.03      0.00     -0.04      0.04   1332.18      1.00
    Rt_transformed_rw_diffs[12]      0.00      0.02      0.00     -0.04      0.04   2042.44      1.00
    Rt_transformed_rw_diffs[13]     -0.00      0.03     -0.00     -0.04      0.04   1278.69      1.00
    Rt_transformed_rw_diffs[14]     -0.00      0.03     -0.00     -0.04      0.04   1777.71      1.00
    Rt_transformed_rw_diffs[15]      0.00      0.03      0.00     -0.04      0.04   1683.29      1.00
    Rt_transformed_rw_diffs[16]     -0.00      0.02     -0.00     -0.03      0.04   1383.06      1.00
    Rt_transformed_rw_diffs[17]     -0.00      0.03     -0.00     -0.04      0.04   1581.60      1.00
    Rt_transformed_rw_diffs[18]      0.00      0.03      0.00     -0.04      0.04   1576.48      1.00
    Rt_transformed_rw_diffs[19]      0.00      0.03      0.00     -0.04      0.04   2169.46      1.00
    Rt_transformed_rw_diffs[20]     -0.00      0.02     -0.00     -0.04      0.04   2041.61      1.00
    Rt_transformed_rw_diffs[21]      0.00      0.02      0.00     -0.03      0.05   1861.29      1.00
    Rt_transformed_rw_diffs[22]      0.00      0.03     -0.00     -0.04      0.04   1796.97      1.00
    Rt_transformed_rw_diffs[23]     -0.00      0.02      0.00     -0.04      0.04   1542.73      1.00
    Rt_transformed_rw_diffs[24]      0.00      0.03      0.00     -0.04      0.04   1625.01      1.00
    Rt_transformed_rw_diffs[25]     -0.00      0.02     -0.00     -0.05      0.04   1711.34      1.00
    Rt_transformed_rw_diffs[26]      0.00      0.03      0.00     -0.04      0.04   1496.53      1.00
    Rt_transformed_rw_diffs[27]     -0.00      0.03     -0.00     -0.05      0.04   1411.17      1.00
    Rt_transformed_rw_diffs[28]     -0.00      0.02      0.00     -0.04      0.04   1478.24      1.00
    Rt_transformed_rw_diffs[29]     -0.00      0.03     -0.00     -0.04      0.04   1557.55      1.00

    Number of divergences: 0

``` python
from pyrenew.mcmcutils import spread_draws
samps = spread_draws(hospmodel.mcmc.get_samples(), [("Rt", "time")])
```

``` python
import numpy as np
import polars as pl
fig, ax = plt.subplots(figsize=[4, 5])

ax.plot(x[0])
samp_ids = np.random.randint(size=25, low=0, high=999)
for samp_id in samp_ids:
    sub_samps = samps.filter(pl.col("draw") == samp_id).sort(pl.col('time'))
    ax.plot(sub_samps.select("time").to_numpy(),
            sub_samps.select("Rt").to_numpy(), color="darkblue", alpha=0.1)
ax.set_ylim([0.4, 1/.4])
ax.set_yticks([0.5, 1, 2])
ax.set_yscale("log")
```

<img
src="pyrenew_demo_files/figure-commonmark/fig-sampled-rt-output-1.png"
id="fig-sampled-rt" />
