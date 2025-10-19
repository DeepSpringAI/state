# Experiment #2: Mixture-Kernel MMD - Ready to Run! ✅

## What You Have Now

You now have an updated `run_experiment_02_competition.sh` that:
- ✅ Uses the **same competition data** as Experiment #1
- ✅ Uses **same batch size (32)** as Experiment #1
- ✅ Tests **Mixture-Kernel MMD** loss function
- ✅ Optionally can combine **both modifications** (early stopping + mixture MMD)

---

## Quick Start

```bash
cd /home/agent/workspace/state_modelling/state
./run_experiment_02_competition.sh
```

---

## Comparison: Mod01 vs Mod02

| Parameter | Mod01 (Early Stopping) | Mod02 (Mixture MMD) | Status |
|-----------|------------------------|---------------------|---------|
| **Data Config** | `competition_support_set/starter.toml` | `competition_support_set/starter.toml` | ✅ Same |
| **Batch Size** | 32 | 32 | ✅ Same |
| **Workers** | 8 | 8 | ✅ Same |
| **Max Steps** | 40,000 | 40,000 | ✅ Same |
| **Checkpoints** | Every 20,000 | Every 20,000 | ✅ Same |
| **Model** | state_sm | state_sm | ✅ Same |
| **Loss Function** | energy (default) | **mixture_mmd** | ⭐ Different |
| **Early Stopping** | ✅ Enabled | ❌ Disabled (optional) | ⭐ Different |
| **Blur Scales** | Single (0.05) | Multi [0.01, 0.05, 0.1, 0.5] | ⭐ Different |

---

## What Each Experiment Tests

### Experiment #1: Early Stopping
**Hypothesis:** Model overfits after optimal point  
**Solution:** Stop training when validation loss plateaus  
**Expected:** 10-15% improvement  
**Tests:** Does early stopping prevent overfitting?

### Experiment #2: Mixture-Kernel MMD
**Hypothesis:** Single-scale distance metric fails with varying data volumes  
**Solution:** Use multi-scale kernels (fine to coarse)  
**Expected:** 8-12% improvement  
**Tests:** Does multi-scale loss handle heterogeneous data better?

---

## Three Ways to Run This

### Option 1: Test Mixture-Kernel MMD Alone (Default)
```bash
./run_experiment_02_competition.sh
```
- Tests if multi-scale loss improves over default energy loss
- Runs full 40,000 steps (no early stopping)
- Compare results to mod01 to see which modification helps more

### Option 2: Combine Both Modifications
Edit line 23 in the script:
```bash
USE_EARLY_STOPPING=true  # Change from false to true
```
Then run:
```bash
./run_experiment_02_competition.sh
```
- Tests if both modifications work well together
- Should give best results (15-25% improvement expected)
- Use this after testing each modification independently

### Option 3: Custom Configuration
You can edit these variables in the script:
```bash
MAX_STEPS=40000          # Training duration
BATCH_SIZE=32            # Batch size
USE_EARLY_STOPPING=false # Combine modifications
PATIENCE=5               # Early stopping patience
```

---

## Expected Output

```
experiments/mod02_mixture_mmd_competition/
├── experiment_info.txt              # Configuration & metadata
├── training.log                     # Full training output
└── mod02_mixture_mmd_competition_model/
    ├── checkpoints/
    │   ├── final.ckpt              # Best checkpoint
    │   ├── last.ckpt               # Latest checkpoint
    │   └── step=*.ckpt             # Periodic saves
    ├── version_0/
    │   └── metrics.csv             # Training metrics
    └── config.yaml                 # Full config
```

---

## Verifying Implementation ✅

All required components are in place:

1. ✅ **Loss Implementation:** `src/state/tx/models/mixture_kernel_mmd.py` exists
2. ✅ **Integration:** Imported in `state_transition.py` (line 16)
3. ✅ **Loss Selection:** Configured in `state_transition.py` (line 181)
4. ✅ **Script:** `run_experiment_02_competition.sh` ready

---

## How to Compare Results

After both experiments complete:

```bash
# Quick comparison
./compare_experiments.sh

# Manual comparison
echo "=== Mod01 (Early Stopping) ==="
tail -2 experiments/mod01_early_stopping_competition/*/version_0/metrics.csv

echo "=== Mod02 (Mixture MMD) ==="
tail -2 experiments/mod02_mixture_mmd_competition/*/version_0/metrics.csv
```

Look for:
- **Lower val_loss** = Better generalization
- **More stable training** = Less variance in loss
- **Better test performance** = Better leaderboard ranking

---

## Next Steps

1. **Run Experiment #2:**
   ```bash
   ./run_experiment_02_competition.sh
   ```

2. **Compare Results:**
   - Which modification helped more?
   - Did val_loss improve vs mod01?

3. **If Both Help:**
   - Enable `USE_EARLY_STOPPING=true` 
   - Run combined experiment (mod03)
   - Expect 15-25% total improvement

4. **Submit to Leaderboard:**
   - Use the best model from experiments
   - Generate predictions
   - Submit to competition

---

## Troubleshooting

### If Training Fails with Loss Error:
```bash
# Verify loss is available
python -c "from state.tx.models.mixture_kernel_mmd import MixtureKernelMMD; print('✅ OK')"
```

### If Import Error:
```bash
# Reinstall package
cd /home/agent/workspace/state_modelling/state
uv sync
```

### If GPU Issues:
```bash
# The script applies gpu_fix_optimized.sh automatically
# If issues persist, run manually:
source gpu_fix_optimized.sh
nvidia-smi  # Check GPU is available
```

---

## Technical Details

### Mixture-Kernel MMD Loss

The loss uses multiple blur scales simultaneously:

```python
blur_scales = [0.01, 0.05, 0.1, 0.5]
# 0.01 = Fine-grained local structure
# 0.05 = Medium-scale patterns
# 0.1  = Larger-scale organization
# 0.5  = Global distribution shape

total_loss = sum([energy_loss(blur=b) for b in blur_scales]) / len(blur_scales)
```

**Why This Helps:**
- Single blur (0.05) misses fine details or global structure
- Multi-scale captures patterns at all levels
- More robust to varying data volumes
- Better for heterogeneous cell populations

### Data Configuration

Both experiments use identical data:
```yaml
Dataset: competition_support_set/starter.toml
  ├── competition_train.h5 (training data)
  ├── k562_gwps.h5, rpe1.h5, jurkat.h5, k562.h5, hepg2.h5
  └── ESM2_pert_features.pt (perturbation embeddings)

Parameters:
  - batch_col: batch_var
  - pert_col: target_gene
  - cell_type_key: cell_type
  - control_pert: non-targeting
  - batch_size: 32
  - num_workers: 8
```

---

**You're ready to run! Good luck! 🚀**

