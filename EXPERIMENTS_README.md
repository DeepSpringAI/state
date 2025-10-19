# Systematic Model Improvement Experiments

## Overview
This directory contains systematic experiments to improve model performance on the competition leaderboard (current rank: ~700-800, target: top 100).

## Quick Start

### Run Experiment #1 (Early Stopping)
```bash
cd /home/agent/workspace/state_modelling/state
source gpu_fix_optimized.sh
./run_experiment_01.sh
```

### Compare All Experiments
```bash
./compare_experiments.sh
```

## Directory Structure
```
experiments/
├── baseline/                      # Original model (no modifications)
├── mod01_early_stopping/         # Modification #1: Early stopping
├── mod02_mixture_kernel_mmd/     # Modification #2: Multi-scale MMD (planned)
├── mod03_single_cell_reg/        # Modification #3: Single-cell regularization (planned)
└── ...                           # Additional modifications
```

Each experiment directory contains:
```
experiment_name/
├── experiment_info.txt           # Configuration and hypothesis
├── training.log                  # Full training output
└── experiment_name_model/        # Model outputs
    ├── checkpoints/              # Saved model weights
    │   ├── last.ckpt
    │   ├── final.ckpt
    │   └── step=XXXX.ckpt
    ├── version_0/
    │   └── metrics.csv           # Training metrics
    └── config.yaml               # Full configuration used
```

## Experiments List

### ✅ Modification #1: Early Stopping
**Status:** Implemented  
**Script:** `run_experiment_01.sh`  
**Expected Impact:** 10-15% improvement  

**What:** Automatically stop training when validation loss stops improving  
**Why:** Prevents overfitting and catastrophic forgetting  
**How:** Monitors val_loss with patience of 5 validation checks

**Files Modified:**
- `src/state/tx/callbacks/early_stopping.py` (new)
- `src/state/tx/callbacks/__init__.py`
- `src/state/_cli/_tx/_train.py`

**Run Command:**
```bash
./run_experiment_01.sh
# or
make train-experiment EXPERIMENT=mod01_early_stopping
```

---

### 📋 Modification #2: Mixture-Kernel MMD
**Status:** Planned  
**Script:** `run_experiment_02.sh` (to be created)  
**Expected Impact:** 8-12% improvement  

**What:** Use multiple kernel scales for distributional loss  
**Why:** Better handles varying cell densities and perturbation scales  
**How:** Replace single blur=0.05 with mixture of scales [0.01, 0.05, 0.1, 0.5]

---

### 📋 Modification #3: Single-Cell MSE Regularization
**Status:** Planned  
**Script:** `run_experiment_03.sh` (to be created)  
**Expected Impact:** 7-10% improvement  

**What:** Add cell-level MSE loss alongside set-level distributional loss  
**Why:** Grounds predictions at individual cell level, not just distribution  
**How:** `total_loss = distributional_loss + lambda * cell_wise_mse`

---

### 📋 Modification #4: Test-Time Augmentation
**Status:** Planned  
**Script:** `run_experiment_04.sh` (to be created)  
**Expected Impact:** 5-10% improvement (inference only!)  

**What:** Sample multiple basal cells, average predictions  
**Why:** Reduces prediction variance, no training cost  
**How:** At inference, predict with 5 different basal cell samples

---

## How to Run Experiments

### Method 1: Use Convenience Scripts
```bash
# Run specific experiment
./run_experiment_01.sh

# Monitor progress
tail -f experiments/mod01_early_stopping/training.log

# Compare results when done
./compare_experiments.sh
```

### Method 2: Use Make Targets
```bash
# Run with default settings
make train-experiment EXPERIMENT=mod01_early_stopping

# Run with custom settings
make train-experiment \
  EXPERIMENT=mod01_custom \
  STEPS=30000 \
  BATCH=128 \
  LR=5e-5
```

### Method 3: Direct Command
```bash
source gpu_fix_optimized.sh
./run.sh tx train \
  data.kwargs.toml_config_path="$(pwd)/examples/mixed.toml" \
  training.max_steps=50000 \
  training.use_early_stopping=true \
  training.early_stopping_patience=5 \
  output_dir="./experiments/my_experiment" \
  name="my_model"
```

## Monitoring Training

### Real-time Monitoring
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training log
tail -f experiments/mod01_early_stopping/training.log

# Watch metrics file
watch -n 5 "tail -5 experiments/mod01_early_stopping/mod01_early_stopping_model/version_0/metrics.csv"
```

### Check Progress
```bash
# View latest metrics
tail -10 experiments/mod01_early_stopping/mod01_early_stopping_model/version_0/metrics.csv | column -t -s,

# Check if early stopping triggered
grep -i "early" experiments/mod01_early_stopping/training.log

# Compare to other experiments
./compare_experiments.sh
```

## Experiment Checklist

For each experiment, document:
- [ ] Hypothesis (what should improve and why)
- [ ] Configuration changes (what's different from baseline)
- [ ] Expected impact (quantitative estimate)
- [ ] Files modified (for reproducibility)
- [ ] Final metrics (train_loss, val_loss, steps)
- [ ] Training time and resource usage
- [ ] Observations (what worked, what didn't)
- [ ] Next steps (continue or abandon this direction)

## Best Practices

### Before Running
1. ✅ Apply GPU fix: `source gpu_fix_optimized.sh`
2. ✅ Check GPU memory: `nvidia-smi`
3. ✅ Verify data: Check `examples/mixed.toml` exists
4. ✅ Set STEPS appropriately (5000 for quick test, 50000 for full run)

### During Training
1. 📊 Monitor val_loss trend (should decrease)
2. 📊 Check gradient_norm (should be stable 0.5-1.5)
3. 📊 Watch GPU memory (should be ~6-7GB)
4. ⚠️ Watch for NaN/Inf values (indicates instability)

### After Training
1. 📈 Compare metrics to baseline
2. 💾 Backup best checkpoint
3. 📝 Update EXPERIMENT_LOG.md
4. 🎯 Decide: continue this direction or try next modification

## Troubleshooting

### Training fails immediately
```bash
# Check config
cat experiments/mod01_early_stopping/mod01_early_stopping_model/config.yaml

# Check linter errors
python -m pylint src/state/tx/callbacks/early_stopping.py

# Verify imports
python -c "from state.tx.callbacks import ImprovedEarlyStopping; print('OK')"
```

### Out of memory
```bash
# Reduce batch size
make train-experiment EXPERIMENT=mod01_early_stopping BATCH=32

# Or reduce cell_set_len (in code)
# Edit: src/state/configs/model/state.yaml
# cell_set_len: 256  # down from 512
```

### Training too slow
```bash
# Use faster GPU fix
source gpu_fix_fast.sh

# Reduce workers
make train-experiment EXPERIMENT=mod01_early_stopping NUM_WORKERS=16

# Reduce logging frequency
# Edit training config: log_every_n_steps: 100
```

### Results worse than baseline
1. Check if model trained long enough (early stop too early?)
2. Verify hyperparameters match baseline
3. Check for bugs in modification
4. Try different hyperparameter values
5. Consider abandoning this modification

## Results Tracking

### Create Results Summary
```bash
# After each experiment
echo "Experiment: mod01_early_stopping" >> RESULTS_SUMMARY.txt
tail -3 experiments/mod01_early_stopping/mod01_early_stopping_model/version_0/metrics.csv >> RESULTS_SUMMARY.txt
echo "" >> RESULTS_SUMMARY.txt
```

### Export for Analysis
```bash
# Collect all metrics.csv files
mkdir -p analysis/
for exp in experiments/*/; do
    exp_name=$(basename $exp)
    cp ${exp}${exp_name}_model/version_0/metrics.csv analysis/${exp_name}_metrics.csv
done

# Analyze in Python/R
python analyze_experiments.py
```

## Next Steps After Experiment #1

1. **If Early Stopping Helps (val_loss improves):**
   - ✅ Keep it enabled for all future experiments
   - ✅ Proceed to Modification #2 (Mixture-Kernel MMD)
   - ✅ Update baseline to include early stopping

2. **If No Improvement:**
   - 🔍 Analyze why (check training curves)
   - 🔍 Try different patience values
   - ⚠️ Consider if this model doesn't overfit (rare)

3. **If Training Stops Too Early:**
   - 🔧 Increase patience from 5 to 7 or 10
   - 🔧 Decrease min_delta from 0.001 to 0.0001
   - 🔧 Check if validation set is too small/noisy

## Contact & Support

For questions or issues:
- Check `EXPERIMENT_LOG.md` for detailed documentation
- Review training logs in `experiments/*/training.log`
- Compare with baseline using `./compare_experiments.sh`

---

**Remember:** One modification at a time! Compare each to baseline before proceeding.


