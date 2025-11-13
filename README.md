# CosmoFieldGen
Tools and scripts to generate cosmological density fields (DM, halo, RSD) from Quijote Latin-Hypercube particle simulations.
This repository provides tools, scripts, and documentation to generate density fields from the lating hypercube sets of Quijote simulation suite. 
It covers the full workflow:

✔ Read Quijote particle snapshots
✔ CIC / PCS mass-assignment to 128³ / 256³ grids
✔ Generate dark matter density fields
✔ Apply redshift-space distortions (RSD)
✔ Generate halo catalogs + voxelize into halo density fields
✔ Normalize fields (ρ/ρ̄ − 1)
✔ Prepare cosmological denisty fields for machine learning models
