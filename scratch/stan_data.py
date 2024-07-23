# numpydoc ignore=GL08
import json

# Load the JSON file
with open("scratch/stan_data_hosp_only.json", "r") as file:
    stan_data = json.load(file)

print(stan_data)


stan_data["i0_over_n_prior_a"][0]
stan_data["i0_over_n_prior_b"][0]
stan_data["state_pop"][0]


for key in sorted(stan_data.keys()):
    print(key)
