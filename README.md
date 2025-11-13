# CosmoFieldGen

**CosmoFieldGen** is a pipeline for generating cosmological density fields from the **Quijote Latin-Hypercube (LH) particle simulations**.  
The Quijote simulation suite provides thousands of large-scale N-body simulations designed for cosmology, emulators, and machine-learning research.  
This repository uses the **2,000-simulation Latin Hypercube ensemble**, where each simulation is initialized with a different random seed and cosmological parameters sampled across a wide range.

We generate both:

- **Initial-condition density fields** (z = 127)  
- **Late-time nonlinear density fields** (z = 0), including dark matter and halo density fields  

You can download the Quijote snapshots directly from the official website:  
🔗 https://quijote-simulations.readthedocs.io/

After downloading the particle snapshots at the desired redshifts (z = 127 and z = 0), use the scripts in this repository to voxelize the data into 3D or 2D density grids at resolutions such as **64³, 128³, 256³, or 512³**.  
The default simulation box size for the Latin Hypercube set is **1000 Mpc/h**.

## 📁 What This Repository Generates

### 1. Generate Dark Matter Density Fields  
Convert particle snapshots into 3D voxel grids using **CIC** or **PCS** mass-assignment schemes.  
Supports multiple resolutions and both **z = 127** (initial condition) and **z = 0** (nonlinear) snapshots.

---

### 2. Generate Redshift-Space Distorted (RSD) Fields  
Apply line-of-sight velocity corrections to particles and reconstruct the corresponding **nonlinear RSD** density fields.

---

### 3. Generate Halo Density Fields  
Read FoF halo catalogs, apply user-defined **mass cuts**, and voxelize halos into 3D density grids using CIC/PCS.  
RSD can be applied to halo positions as well.

---

### 4. Generate 2D Projected Density Fields  
Create 2D slices from 3D density grids using **central-slab projection** with mean/median/max aggregation.  
This functionality is implemented in `Generate_train_2D_samples.py`.

---

### 5. Precompute Velocity-Field Features  
Transform 3D density fields into a **6-channel velocity-feature stack** using FFT-based operations in k-space.  
This computation solves a Poisson-like equation to obtain velocity components and spatial derivatives, providing richer conditioning inputs for ML models.
