# 📊 What is the Default Batch Size?

## ✅ **THE ANSWER:**

**It depends on how you run the training!** There are multiple default values in different places:

| Method | Default Batch Size | Location |
|--------|-------------------|----------|
| **Using `make train`** | **64** | `Makefile` line 4 |
| **Direct Python call** | 16 | `src/state/configs/training/default.yaml` line 3 |
| **DataModule only** | 128 | `cell_load/data_modules/perturbation_dataloader.py` line 39 |
| **SE model config** | 128 | `src/state/configs/state-defaults.yaml` line 209 |

---

## 🔍 **Which One Actually Gets Used?**

### **Scenario 1: Using `make train` (Most Common)**

```bash
make train
# or
make train BATCH=32  # Override to 32
```

**Default batch size: 64**

**Why?** The Makefile passes `BATCH=64` to the training command:
```makefile
BATCH ?= 64
```

This gets passed to the config as `cfg["training"]["batch_size"] = 64`.

---

### **Scenario 2: Using Hydra Config (Direct Python)**

```bash
python src/state/_cli/_tx/_train.py
```

**Default batch size: 16**

**Why?** The training script loads `src/state/configs/training/default.yaml`:
```yaml
batch_size: 16
```

---

### **Scenario 3: Creating DataModule Directly (Code)**

```python
from cell_load.data_modules import PerturbationDataModule

data_module = PerturbationDataModule(
    toml_config_path="examples/mixed.toml"
    # No batch_size specified
)
```

**Default batch size: 128**

**Why?** The `__init__` signature has:
```python
def __init__(
    self,
    toml_config_path: str,
    batch_size: int = 128,  # ← Default here
    ...
):
```

---

## 📝 **Priority Order (What Overrides What)**

When you run training, the batch size is determined by this priority (highest to lowest):

```
1. Command-line argument (BATCH=X in make)
   ↓
2. Config file value (cfg["training"]["batch_size"])
   ↓
3. DataModule __init__ default (128)
```

---

## 🎯 **Examples:**

### **Example 1: Default make train**
```bash
make train
```
→ Uses `BATCH ?= 64` from Makefile
→ **Final batch size: 64**

---

### **Example 2: Override batch size**
```bash
make train BATCH=32
```
→ Overrides Makefile default
→ **Final batch size: 32**

---

### **Example 3: Direct Hydra call**
```bash
python src/state/_cli/_tx/_train.py
```
→ Uses `src/state/configs/training/default.yaml` (batch_size: 16)
→ **Final batch size: 16**

---

### **Example 4: Hydra with override**
```bash
python src/state/_cli/_tx/_train.py training.batch_size=48
```
→ Overrides config value
→ **Final batch size: 48**

---

## 🔧 **How to Check Your Current Batch Size:**

### **Method 1: Check during training**

Look at the printed output when training starts:
```
num_workers: 32
batch size: 64  # ← Shows here (or None if using batch_sampler)
```

### **Method 2: Check first batch shape**

After the fix I suggested earlier, or add this to `_train.py`:
```python
data_module.setup(stage="fit")
dl = data_module.train_dataloader()

# Get actual batch size
for batch in dl:
    print(f"Actual batch size: {batch['ctrl_cell_emb'].shape[0]}")
    break
```

### **Method 3: Check Makefile**

```bash
grep "BATCH" Makefile
```
Output:
```
BATCH         ?= 64
```

### **Method 4: Check config file**

```bash
cat src/state/configs/training/default.yaml | grep batch_size
```
Output:
```
batch_size: 16
```

---

## 💡 **Recommendation:**

**Always specify the batch size explicitly!** This avoids confusion:

```bash
# Good - explicit
make train BATCH=64

# Better - document in your experiment
make train BATCH=64 NAME=my_experiment_batch64
```

Or create a shell script:
```bash
#!/bin/bash
# my_experiment.sh

BATCH_SIZE=64
STEPS=5000
LR=1e-4

make train BATCH=$BATCH_SIZE STEPS=$STEPS LR=$LR NAME=my_experiment
```

---

## 📊 **Summary Table:**

| Question | Answer |
|----------|--------|
| **Default with `make train`?** | **64** (from Makefile) |
| **Default with Hydra?** | **16** (from config) |
| **Default in code?** | **128** (from DataModule) |
| **How to override?** | `make train BATCH=X` or `training.batch_size=X` |
| **How to check?** | Print first batch shape or check sampler |

---

## 🎓 **Why So Many Defaults?**

Different defaults exist because:

1. **Makefile (64)**: Optimized for this specific project/competition
2. **Config (16)**: Conservative default that works on most GPUs
3. **DataModule (128)**: General-purpose default for the library
4. **SE Config (128)**: Default for State Embedding pre-training

The actual default you get depends on your entry point!

---

## ✅ **TL;DR:**

**If you don't pass any batch_size argument:**
- Using `make train`: **Batch size = 64**
- Using Python directly: **Batch size = 16**
- Creating DataModule in code: **Batch size = 128**

**Always check or specify explicitly to avoid confusion!** 🎯


