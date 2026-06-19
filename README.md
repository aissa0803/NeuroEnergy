# NeuroEnergy — AI Model Energy Estimation Tool

A web platform for estimating the energy consumption of PyTorch neural networks on embedded hardware boards (JetsonNano, CoralDevBoard).

---

## Project structure

```
energy_estimation/
├── app.py                          # Streamlit web app (the platform)
├── requirements.txt                # Python dependencies
├── logo/
│   └── logo.png
├── .streamlit/
│   └── config.toml                 # Theme (green accent, light background)
└── energy_estimator/               # Core estimation package
    ├── __init__.py                 # Public API
    ├── power_lookup_tables.py      # Table loading & caching
    ├── layer_energy_interpolation.py  # estimate_energy()
    ├── grid_interpolation.py       # Bilinear grid math
    ├── model_energy_aggregation.py # estimate_model_energy() for nn.Module
    └── data/
        ├── CoralDevBoard/
        │   └── linear_power_report.xlsx
        └── JetsonNano/
            ├── linear_power_report.xlsx
            ├── conv_k3_p0_power_report.xlsx
            ├── conv_k3_p1_power_report.xlsx
            ├── conv_k5_p0_power_report.xlsx
            └── conv_k5_p1_power_report.xlsx
```

---

## How to run

**1. Install uv** (if not already installed)
```bash
pip install uv
```

**2. Create and activate the virtual environment**
```bash
uv venv .venv
```
On Windows:
```bash
.venv\Scripts\activate
```
On macOS/Linux:
```bash
source .venv/bin/activate
```

**3. Install dependencies**
```bash
uv pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app.py
```

---

## How the logic works

### 1. Measured power tables

Energy was measured on real hardware by running specific layer configurations and recording average power. Results are stored as 2D Excel matrices:

- **rows** → input dimension (features / channels)
- **columns** → output dimension (features / channels)
- **cell value** → measured energy in Joules

### 2. Layer key

Each layer type maps to a specific table. For Conv2d, the kernel size and padding select the right file:

| Layer | Key | Table file |
|---|---|---|
| `nn.Linear` | `linear` | `linear_power_report.xlsx` |
| `nn.Conv2d(kernel=3, padding=0)` | `conv_k3_p0` | `conv_k3_p0_power_report.xlsx` |
| `nn.Conv2d(kernel=3, padding=1)` | `conv_k3_p1` | `conv_k3_p1_power_report.xlsx` |
| `nn.Conv2d(kernel=5, padding=0)` | `conv_k5_p0` | `conv_k5_p0_power_report.xlsx` |
| `nn.Conv2d(kernel=5, padding=1)` | `conv_k5_p1` | `conv_k5_p1_power_report.xlsx` |

CoralDevBoard only has linear measurements; JetsonNano has both linear and Conv2d.

### 3. Bilinear interpolation

Tables only contain measurements at discrete sizes (e.g. 64, 128, 256 …). For any arbitrary size, the package estimates energy by interpolating over the 4 nearest measured points:

```
Find the 4 surrounding points:
  (in_lo, out_lo) → Q11     (in_lo, out_hi) → Q12
  (in_hi, out_lo) → Q21     (in_hi, out_hi) → Q22

Compute fractional positions:
  tx = (in_dim  - in_lo)  / (in_hi  - in_lo)
  ty = (out_dim - out_lo) / (out_hi - out_lo)

Interpolate:
  R1 = Q11*(1-ty) + Q12*ty      ← along output axis, low input
  R2 = Q21*(1-ty) + Q22*ty      ← along output axis, high input
  E  = R1*(1-tx)  + R2*tx       ← along input axis
```

Values outside the measured range are clamped to the nearest edge. Each table is also anchored at (0, 0) → 0 J so fully-pruned layers correctly return zero energy.

### 4. Pruning

When a pruning rate `p` is applied to a layer, the effective output dimension is reduced and the energy is re-interpolated — not just scaled. This is more accurate because energy growth is not linear with dimension.

```
eff_out = out_dim × (1 - p)

E_dense = estimate_energy(board, key, in_dim, out_dim)
E_used  = estimate_energy(board, key, in_dim, eff_out)
```

### 5. Total energy

All per-layer energies are summed to produce the total estimate. The package also supports **differentiable pruning** via soft masks (`mask.sum()` as the effective output dim), allowing gradients to flow back through the energy estimate during training.

---

## Package API

```python
from energy_estimator import list_boards, list_layer_keys, load_table, estimate_model_energy
from energy_estimator.layer_energy_interpolation import estimate_energy

# Discover available hardware
list_boards()                        # ["CoralDevBoard", "JetsonNano"]
list_layer_keys("JetsonNano")        # ["conv_k3_p0", "conv_k3_p1", ..., "linear"]

# Estimate a single layer
energy = estimate_energy("JetsonNano", "linear", in_dim=512, out_dim=256)

# Estimate a full PyTorch model
results = estimate_model_energy(model, board="JetsonNano")
print(results["total_energy"])       # scalar tensor in Joules
print(results["layers"])             # per-layer breakdown
```

---

## Platform features

| Feature | Description |
|---|---|
| Manual builder | Add Linear / Conv2d layers one by one |
| Model upload | Auto-extract layers from a `.pt` / `.pth` state dict |
| Board selector | Switch between available hardware boards |
| Pruning per layer | Set a pruning rate 0–1 per layer |
| Unit toggle | Display results in Joules or millijoules |
| Per-layer bar chart | Energy breakdown across layers |
| Cumulative line chart | Running total energy through the network |
| Layer details table | Full breakdown with dense vs. used energy |
