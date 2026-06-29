# SMPL LBS Lab — Linear Blend Skinning Visualization

> A step-by-step dissection and visualization of the SMPL Linear Blend Skinning pipeline, with manual reimplementation and official forward-pass verification.

---

## Overview

This project implements a complete **manual LBS (Linear Blend Skinning)** pipeline based on the [SMPL](https://smpl.is.tue.mpg.de/) parametric human body model. It extracts every intermediate quantity from the official `lbs()` function and renders them as 3D visualizations at four key stages of the pipeline. The manual implementation is then numerically verified against the official SMPL forward pass.

## Objectives

1. Understand the relationships among the template mesh, shape parameters (`β`), pose parameters (`θ`), joint regressor, and skinning weights in a parametric body model.
2. Internalize the four stages of LBS:
   - **(a)** Template mesh `T̄` with skinning weights `W`
   - **(b)** Shape-corrected mesh `T̄ + B_S(β)` with regressed joints `J(β)`
   - **(c)** Pose-corrected mesh `T_P(β,θ) = T̄ + B_S(β) + B_P(θ)`
   - **(d)** Final skinned result after LBS
3. Call the SMPL model and manually extract + visualize the key intermediate tensors that the official `lbs()` implementation produces internally.

---

## Core Architecture: The 4-Stage LBS Pipeline

### Stage (a) — Template Mesh & Skinning Weights

```
T̄  (N_V × 3)         →  template vertices in rest pose (T-pose)
W  (N_V × K)          →  skinning weight of each vertex w.r.t. each joint
```

- The template `T̄` is the neutral human mesh in T-pose (6890 vertices, 13776 faces).
- `W` is a matrix of per-vertex, per-joint influence weights (24 joints for SMPL). Each row sums to 1.
- At this point the mesh has **not** adopted any body shape or pose — but every vertex already knows which skeletal joints it will follow.

### Stage (b) — Shape Blend & Joint Regression

```
v_shaped = T̄ + B_S(β)                     // β ∈ R^{10}: shape coefficients
J        = J_regressor(v_shaped)           // 24 joints regressed from shaped mesh
```

- `β` encodes body identity (height, weight, shoulder width, etc.) via a linear blend shape basis (`shapedirs`).
- `B_S(β) = blend_shapes(β, shapedirs)` adds these displacements to the template.
- Joint positions `J` are **not** constants — they are regressed from the shaped vertices via `vertices2joints(J_regressor, v_shaped)`. A heavier person's hip joints sit at different world positions than a thinner person's.

### Stage (c) — Pose Blend Shapes

```
full_pose     = cat(global_orient, body_pose)      // [1, 72]  axis-angle
rot_mats      = batch_rodrigues(full_pose)          // [1, 24, 3, 3]
pose_feature  = flatten(rot_mats[:, 1:, :, :] − I)  // deviation from identity
pose_offsets  = matmul(pose_feature, posedirs)       // [1, N_V, 3]
v_posed       = v_shaped + pose_offsets
```

- Before binding vertices to bones, SMPL adds **pose-dependent corrective offsets** (`B_P(θ)`) to handle non-rigid deformations near bending joints (shoulders, elbows, knees).
- `pose_feature` is the deviation of each joint's rotation matrix from the identity — this encodes *how much* each joint is bent.
- `posedirs` maps these deviations to vertex displacements via a linear model.
- **v_posed is NOT the final result** — the mesh has been deformed, but has not yet been bound to the skeleton.

### Stage (d) — Linear Blend Skinning

```
J_transformed, A = batch_rigid_transform(rot_mats, J, parents)
T                 = matmul(W, A)                     // per-vertex 4×4 transform
v_homo            = [v_posed; 1]                     // homogeneous coordinates
verts             = T · v_homo                        // final skinned vertices
```

- `batch_rigid_transform` walks the kinematic tree (defined by `parents`) to compute the **global** rigid transform `G_k` for each joint.
- Each vertex's final transform `T_i` is a *weighted blend* of the global transforms of the joints that influence it:
  ```
  v_i′ = Σ_{k=1}^{K}  w_{ik} · G_k · [v_i^{posed}; 1]
  ```
- This is why it is called **Linear Blend Skinning**: each vertex does not follow a single bone, but blends across multiple bone transforms with weighted averaging.

### Five Core Variables

| Variable | Shape | Meaning |
|---|---|---|
| `v_template` | [N_V, 3] | Raw template mesh in T-pose |
| `v_shaped` | [N_V, 3] | Template + shape-dependent deformation |
| `J` | [K, 3] | Joint positions regressed from v_shaped |
| `v_posed` | [N_V, 3] | v_shaped + pose-dependent corrective offsets |
| `verts` | [N_V, 3] | Final skinned mesh after LBS |

---

## Project Structure

```
Tzz8/
├── run_lbs_lab.py                       # Main script (manual LBS + visualization + verification)
├── outputs/                             # All generated outputs
│   ├── stage_a_template_weights.png     # Template mesh with joint-weight heatmap
│   ├── all_joint_weights.png            # Per-face dominant-joint coloring (optional)
│   ├── stage_b_shaped_joints.png        # Shape-corrected mesh + regressed joints
│   ├── stage_c_pose_offsets.png         # Pose-corrected mesh (colored by offset magnitude)
│   ├── stage_d_lbs_result.png           # Final skinned mesh + transformed joints
│   ├── comparison_grid.png              # 2×2 grid comparing all four stages
│   └── summary.txt                      # Model metadata + numerical verification
├── .gitignore
└── README.md
```

> **Note:** The SMPL model file (`SMPL_NEUTRAL.pkl`) is **not** included in this repository. Download it from [smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de/).

---

## Dependencies

```bash
pip install torch numpy matplotlib smplx
```

| Package | Purpose |
|---|---|
| `torch` | Tensor computation |
| `numpy` | Numerical operations |
| `matplotlib` | 3D mesh rendering |
| `smplx` | SMPL model loading and official forward pass |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yaqingtang666-lab/Tzz8.git
cd Tzz8

# 2. Install dependencies
pip install torch numpy matplotlib smplx

# 3. Place SMPL_NEUTRAL.pkl under ./models/smpl/

# 4. Run the lab
python run_lbs_lab.py --model-dir ./models --out-dir ./outputs --joint-id 18 --num-betas 10
```

### Command-Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--model-dir` | `./models` | Directory containing `smpl/SMPL_NEUTRAL.pkl` |
| `--out-dir` | `./outputs` | Output directory for images and summary |
| `--joint-id` | `18` | Joint index to visualize in stage (a) heatmap |
| `--num-betas` | `10` | Number of shape parameters to use |

---

## Verification Results

The manual LBS implementation was verified against the official SMPL forward pass with identical inputs:

```
manual_vs_official_mean_abs_error: 0.0000000000
manual_vs_official_max_abs_error:  0.0000000000
```

Both errors are zero to the precision of `float32`, confirming that the hand-written pipeline is **numerically identical** to the official `smplx` implementation.

### Model Statistics

| Property | Value |
|---|---|
| Vertices | 6890 |
| Faces | 13776 |
| Joints | 24 |
| Shape params (`β`) | 10 |

---

## Key Design Decisions

1. **`_ChumpyArrayShim`** — A minimal pickle compatibility shim that allows loading legacy SMPL `.pkl` files *without* installing the deprecated `chumpy` package. This works by registering a fake `chumpy.ch` module that reconstructs numpy arrays from the pickled object's internal state.

2. **`prepare_posedirs`** — Some SMPL variants store `posedirs` in transposed shapes (e.g., `[V×3, P]` instead of `[P, V×3]`). This helper detects the shape and transposes if needed to ensure compatibility with the official `lbs()` convention.

3. **Coordinate system transform** — SMPL uses a Y-up coordinate system, while matplotlib uses Z-up. The helper `smpl_to_plot_coords` swaps Y and Z axes for correct 3D rendering.

---

## Output Visualizations

| File | Description |
|---|---|
| `stage_a_template_weights.png` | T-pose mesh with per-vertex heatmap showing skinning weight for a single joint |
| `all_joint_weights.png` | Each face colored by its dominant (max-weight) joint — shows global joint influence regions |
| `stage_b_shaped_joints.png` | Body shape changed via `β`; white dots are regressed joint positions |
| `stage_c_pose_offsets.png` | Pose-corrected mesh colored by `|pose_offsets|` — hotter colors = larger corrective displacements near bending joints |
| `stage_d_lbs_result.png` | Final skinned mesh in target pose with transformed joints |
| `comparison_grid.png` | 2×2 overview of all four pipeline stages side by side |

---

## References

- Loper, M., Mahmood, N., Romero, J., Pons-Moll, G., & Black, M. J. (2015). **SMPL: A Skinned Multi-Person Linear Model**. *ACM Transactions on Graphics (TOG)*, 34(6), 248.
- Official SMPL website: [https://smpl.is.tue.mpg.de/](https://smpl.is.tue.mpg.de/)
- SMPL-X Python library: [https://github.com/vchoutas/smplx](https://github.com/vchoutas/smplx)
