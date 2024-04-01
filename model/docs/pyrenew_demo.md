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

``` python
np.random.seed(3312)
q = SimpleRandomWalkProcess(dist.Normal(0, 0.001))
with seed(rng_seed=np.random.randint(0,1000)):
    q_samp = q.sample(duration=100)

plt.plot(np.exp(q_samp[0]))
```

![](pyrenew_demo_files/figure-commonmark/fig-randwalk-output-1.png)

``` python
from pyrenew.latent import Infections, HospitalAdmissions, Infections0
from pyrenew.observation import PoissonObservation
from pyrenew.deterministic import DeterministicPMF, DeterministicVariable
from pyrenew.model import HospitalizationsModel
from pyrenew.process import RtRandomWalkProcess

# Initializing model components:

# A deterministic generation time
gen_int = DeterministicPMF(
    (jnp.array([0.25, 0.25, 0.25, 0.25]),),
    )

# Baseline infections
I0 = Infections0()

# The latent infections process
latent_infections = Infections()

# A deterministic infection to hosp pmf
inf_hosp_int = DeterministicPMF(
    (jnp.array([0, 0, 0,0,0,0,0,0,0,0,0,0,0, 0.25, 0.5, 0.1, 0.1, 0.05]),),
    )

# The latent hospitalization process
latent_hospitalizations = HospitalAdmissions(inform_hosp=inf_hosp_int)

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
    gen_int=gen_int,
    I0=I0,
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
           1.1837622], dtype=float32), infections=Array([0.05214045, 0.06867922, 0.08761451, 0.11476436, 0.09757317,
           0.10547114, 0.1167062 , 0.13010225, 0.13824694, 0.14372033,
           0.15924728, 0.17601486, 0.19236736, 0.21483542, 0.23664482,
           0.25865382, 0.27503362, 0.30029488, 0.3289544 , 0.35262382,
           0.37418258, 0.41274938, 0.43839005, 0.47672123, 0.50913286,
           0.5625195 , 0.6113282 , 0.67092246, 0.7138808 , 0.77217466,
           0.819254  ], dtype=float32), IHR=Array(0.04929917, dtype=float32), latent=Array([0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.00064262, 0.0021317 ,
           0.00302979, 0.00416974, 0.0049305 , 0.00487205, 0.00530097,
           0.00576412, 0.00624666, 0.00665578, 0.00711595, 0.0078055 ,
           0.00854396, 0.00939666, 0.01042083, 0.0114624 , 0.01246538,
           0.01345188], dtype=float32), sampled=Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=int32))

``` python
fig, ax = plt.subplots(nrows=3, sharex=True)
ax[0].plot(x.infections)
ax[0].set_ylim([1/5, 5])
ax[1].plot(x.latent)
ax[2].plot(x.sampled, 'o')
for axis in ax[:-1]:
    axis.set_yscale("log")
```

![](pyrenew_demo_files/figure-commonmark/fig-hosp-output-1.png)

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
                             I0      1.27      1.10      0.97      0.10      2.42   1132.34      1.00
                            IHR      0.05      0.00      0.05      0.05      0.05   2306.45      1.00
                            Rt0      1.23      0.17      1.23      0.93      1.48   1327.22      1.00
     Rt_transformed_rw_diffs[0]     -0.00      0.02     -0.00     -0.04      0.04   1404.95      1.00
     Rt_transformed_rw_diffs[1]      0.00      0.03      0.00     -0.04      0.04   2280.86      1.00
     Rt_transformed_rw_diffs[2]     -0.00      0.02     -0.00     -0.04      0.04   2119.83      1.00
     Rt_transformed_rw_diffs[3]      0.00      0.02     -0.00     -0.04      0.04   2196.86      1.00
     Rt_transformed_rw_diffs[4]      0.00      0.02     -0.00     -0.03      0.04   2391.45      1.00
     Rt_transformed_rw_diffs[5]      0.00      0.03      0.00     -0.04      0.04   2043.02      1.00
     Rt_transformed_rw_diffs[6]      0.00      0.02      0.00     -0.04      0.04   1514.40      1.00
     Rt_transformed_rw_diffs[7]     -0.00      0.02     -0.00     -0.04      0.04   2619.69      1.00
     Rt_transformed_rw_diffs[8]      0.00      0.03      0.00     -0.04      0.04   1883.84      1.00
     Rt_transformed_rw_diffs[9]      0.00      0.03      0.00     -0.04      0.04   2015.66      1.00
    Rt_transformed_rw_diffs[10]      0.00      0.02      0.00     -0.04      0.04   2045.47      1.00
    Rt_transformed_rw_diffs[11]     -0.00      0.03      0.00     -0.04      0.04   1615.10      1.00
    Rt_transformed_rw_diffs[12]      0.00      0.02      0.00     -0.04      0.04   2206.32      1.00
    Rt_transformed_rw_diffs[13]      0.00      0.03      0.00     -0.04      0.04   1175.93      1.00
    Rt_transformed_rw_diffs[14]     -0.00      0.03     -0.00     -0.04      0.04   1606.26      1.00
    Rt_transformed_rw_diffs[15]     -0.00      0.03     -0.00     -0.04      0.04   2344.62      1.00
    Rt_transformed_rw_diffs[16]     -0.00      0.02      0.00     -0.04      0.04   1522.33      1.00
    Rt_transformed_rw_diffs[17]      0.00      0.03      0.00     -0.04      0.04   2157.17      1.00
    Rt_transformed_rw_diffs[18]     -0.00      0.02     -0.00     -0.04      0.04   1594.95      1.00
    Rt_transformed_rw_diffs[19]      0.00      0.03     -0.00     -0.04      0.04   1698.70      1.00
    Rt_transformed_rw_diffs[20]      0.00      0.02      0.00     -0.04      0.04   1726.18      1.00
    Rt_transformed_rw_diffs[21]      0.00      0.02     -0.00     -0.04      0.04   2386.35      1.00
    Rt_transformed_rw_diffs[22]      0.00      0.03      0.00     -0.04      0.04   2028.63      1.00
    Rt_transformed_rw_diffs[23]      0.00      0.02      0.00     -0.04      0.03   1669.71      1.00
    Rt_transformed_rw_diffs[24]      0.00      0.02      0.00     -0.04      0.04   2126.33      1.00
    Rt_transformed_rw_diffs[25]     -0.00      0.02     -0.00     -0.04      0.04   2119.74      1.00
    Rt_transformed_rw_diffs[26]      0.00      0.03      0.00     -0.04      0.04   2657.91      1.00
    Rt_transformed_rw_diffs[27]     -0.00      0.03      0.00     -0.04      0.04   1939.30      1.00
    Rt_transformed_rw_diffs[28]     -0.00      0.02     -0.00     -0.04      0.04   1737.84      1.00
    Rt_transformed_rw_diffs[29]     -0.00      0.03     -0.00     -0.04      0.04   2105.55      1.00

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

![](pyrenew_demo_files/figure-commonmark/fig-sampled-rt-output-1.png)
