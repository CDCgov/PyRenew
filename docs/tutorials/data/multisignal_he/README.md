# Multisignal H+E tutorial artifacts

This directory stores precomputed fit artifacts for
`docs/tutorials/multisignal_H_E_model.qmd`.

Regenerate the artifacts from the repository root with:

```bash
uv run python docs_scripts/generate_multisignal_he_artifacts.py
```

The generator must be run on a machine that can fit the PyRenew tutorial
models, including the linked-ascertainment variant, and the custom NumPyro H+E
model. The documentation build reads the generated `npz`, `csv`, and `json`
files from this directory instead of running the fits during Quarto rendering.
The custom-H-E artifacts include reduced scalar draws and latent infection
draws for plotting posterior infection uncertainty.
