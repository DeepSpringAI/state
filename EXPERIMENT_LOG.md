# Systematic Experiment Log

## Competition Improvement Experiments
**Goal:** Improve leaderboard ranking from ~700-800 to top 100
**Baseline:** Previous model (temp_zorba, mixed_zorba)

---

## Modification #1: Early Stopping

**Date:** 2025-10-12
**Status:** ✅ Implemented, Ready to Test

### **Problem Identified**
From user notes:
> "early stop should be there, running for more epochs cuz forgetting in the scenario"

The model was training too long and experiencing catastrophic forgetting - learning stopped improving but training continued, causing overfitting.

### **Solution Implemented**
Added `ImprovedEarlyStopping` callback that:
- Monitors `val_loss` during validation
- Stops training if no improvement for `patience` validation checks
- Default patience: 5 (stops after 5 * 2000 = 10,000 steps without improvement)
- Minimum delta: 0.001 (must improve by at least 0.1% to count)
- Verbose logging of best metrics

### **Files Modified**
1. **Created:** `src/state/tx/callbacks/early_stopping.py`
   - New callback class with enhanced features
   
2. **Modified:** `src/state/tx/callbacks/__init__.py`
   - Exported `ImprovedEarlyStopping`
   
3. **Modified:** `src/state/_cli/_tx/_train.py` (lines 212-224)
   - Added early stopping to training loop
   - Controlled by config: `training.use_early_stopping` (default: True)
   - Configurable patience: `training.early_stopping_patience` (default: 5)
   - Configurable min_delta: `training.early_stopping_min_delta` (default: 0.001)

4. **Modified:** `Makefile`
   - Added `EXPERIMENT`, `EXP_OUTPUT`, `EXP_NAME` variables
   - Added `train-experiment` target for systematic testing
   - Creates separate output directories per experiment

### **Configuration**
Early stopping is **enabled by default**. To configure:

```yaml
# In config or command line:
training:
  use_early_stopping: true
  early_stopping_patience: 5        # Number of validation checks
  early_stopping_min_delta: 0.001   # Minimum improvement (0.1%)
```

### **How to Run This Experiment**

```bash
# Apply GPU fix
source gpu_fix_optimized.sh

# Run experiment
make train-experiment EXPERIMENT=mod01_early_stopping STEPS=50000

# Or with custom patience
./run.sh tx train \
  data.kwargs.toml_config_path="$(pwd)/examples/mixed.toml" \
  training.max_steps=50000 \
  training.early_stopping_patience=5 \
  output_dir="./experiments/mod01_early_stopping" \
  name="mod01_early_stopping"
```

### **Output Location**
- **Directory:** `experiments/mod01_early_stopping/mod01_early_stopping_model/`
- **Checkpoints:** `experiments/mod01_early_stopping/mod01_early_stopping_model/checkpoints/`
- **Metrics:** `experiments/mod01_early_stopping/mod01_early_stopping_model/version_0/metrics.csv`
- **Config:** `experiments/mod01_early_stopping/mod01_early_stopping_model/config.yaml`

### **Expected Results**
- Training will stop automatically when validation loss plateaus
- Likely stops around 15,000-25,000 steps (vs full 50,000)
- Better generalization on test set
- Prevents "forgetting" phenomenon

### **Success Criteria**
- ✅ val_loss stops decreasing → early stop triggers
- ✅ Final model performs better on holdout data
- ✅ Training time reduced (saves compute!)
- 🎯 Expected improvement: **10-15% better ranking**

### **Monitoring**
Watch for these messages during training:
```
[EarlyStopping] Current val_loss: 0.285000, Best: 0.276000, Patience: 0/5
[EarlyStopping] Current val_loss: 0.287000, Best: 0.276000, Patience: 1/5
...
[EarlyStopping] Current val_loss: 0.289000, Best: 0.276000, Patience: 5/5
Training stopped early at step 22000 due to no improvement in val_loss
```

### **Comparison to Baseline**
Will compare against:
- Baseline (no early stopping, full 50k steps)
- Previous competition submissions (temp_zorba, mixed_zorba)

---

---

## Modification #2: Mixture-Kernel MMD

**Date:** 2025-10-12
**Status:** ✅ Implemented, Ready to Test (Running in Parallel)

### **Problem Identified**
From user notes:
> "How does the distance metric work when data volume is increased? the model fails on the leaderboard."

Single-scale MMD (single blur parameter) fails when:
- Data volume varies across batches
- Cell densities are heterogeneous
- Perturbations have multi-scale effects (local + global)

### **Solution Implemented**
Added three variants of mixture-kernel MMD:

1. **MixtureKernelMMD**: Fixed multi-scale loss
   - Uses blur scales: [0.01, 0.05, 0.1, 0.5]
   - Equal weighting of all scales
   - Captures fine → coarse structure

2. **AdaptiveMixtureKernelMMD**: Adaptive scale weighting
   - Adapts based on batch_size, variance, or curriculum
   - Smart weight adjustment during training
   - More sophisticated than fixed weights

3. **HierarchicalMMD**: Coarse-to-fine hierarchy
   - Mirrors biological pathway effects
   - Hierarchical weights: [0.5, 0.3, 0.2] (coarse → fine)
   - More weight on easier-to-match coarse scales

### **Files Modified**
1. **Created:** `src/state/tx/models/mixture_kernel_mmd.py`
   - Three new loss classes
   - Multi-scale kernel MMD implementation
   
2. **Modified:** `src/state/tx/models/state_transition.py` (lines 180-191)
   - Added loss selection for `mixture_mmd`, `adaptive_mmd`, `hierarchical_mmd`
   - Integrated with existing loss framework

3. **Created:** `run_experiment_02.sh`
   - Automated experiment runner for mod #2

### **Configuration**
To use mixture-kernel MMD:

```yaml
# In config or command line:
model:
  kwargs:
    loss: mixture_mmd  # or adaptive_mmd or hierarchical_mmd
    blur_scales: [0.01, 0.05, 0.1, 0.5]  # optional, customize scales
```

### **How to Run This Experiment**

```bash
# Run in parallel with mod01 (don't stop mod01!)
./run_experiment_02.sh

# Or manually
./run.sh tx train \
  model.kwargs.loss=mixture_mmd \
  training.max_steps=5000 \
  output_dir="./experiments/mod02_mixture_kernel_mmd" \
  name="mod02_mixture_kernel_mmd_model"
```

### **Output Location**
- **Directory:** `experiments/mod02_mixture_kernel_mmd/mod02_mixture_kernel_mmd_model/`
- **Checkpoints:** `experiments/mod02_mixture_kernel_mmd/mod02_mixture_kernel_mmd_model/checkpoints/`
- **Metrics:** `experiments/mod02_mixture_kernel_mmd/mod02_mixture_kernel_mmd_model/version_0/metrics.csv`

### **Expected Results**
- Better handling of varying batch sizes
- More robust distance metric
- Improved generalization on heterogeneous data
- Better captures multi-scale perturbation effects

### **Success Criteria**
- ✅ val_loss lower than baseline (mod01)
- ✅ More stable training (less variance in loss)
- ✅ Better performance on varying data volumes
- 🎯 Expected improvement: **8-12% better ranking**

### **Comparison to Baseline**
Will compare against:
- Baseline (single-scale energy loss)
- Mod01 (early stopping only)

---

## Next Modifications (Planned)

### Modification #3: Single-Cell MSE Regularization  
**Status:** 📋 Planned
**Files to modify:** `src/state/tx/models/state_transition.py`
**Expected improvement:** 7-10%

### Modification #3: Single-Cell MSE Regularization  
**Status:** 📋 Planned
**Files to modify:** `src/state/tx/models/state_transition.py`
**Expected improvement:** 7-10%

### Modification #4: Test-Time Augmentation
**Status:** 📋 Planned
**Files to create:** `src/state/_cli/_tx/_predict.py` enhancement
**Expected improvement:** 5-10% (FREE - inference only!)

---

## Experiment Tracking Template

For each experiment, record:
- [ ] Final val_loss
- [ ] Number of steps trained
- [ ] Training time
- [ ] GPU memory usage
- [ ] Test set performance (if available)
- [ ] Leaderboard rank (if submitted)

---

## Notes
- Keep all experiment outputs separate
- Document all hyperparameter changes
- Save best model checkpoints for comparison
- Track computational cost (FLOPS, time, memory)

