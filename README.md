# EnsembleSVR-XACLE
[![XACLE_Dataset](https://img.shields.io/badge/GitHub-XACLE-blue)](https://github.com/XACLE-Challenge/the_first_XACLE_challenge_baseline_model)

# Installation
### 1. Clone the repository
```bash
git clone https://github.com/Snehitc/EnsembleSVR-XACLE.git
```

```bash
cd EnsembleSVR-XACLE
```

### 2. Create environment
```bash
conda create -n EnsembleSVR python=3.9
```
```bash
conda activate EnsembleSVR
```

### 3. Install requirements
```bash
pip install -r requirements.txt
```

### 4. Install Torch
```bash
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4
```

### 5. Download Dataset: [![XACLE_Dataset](https://img.shields.io/badge/GitHub-XACLE-blue)](https://github.com/XACLE-Challenge/the_first_XACLE_challenge_baseline_model)
Refer to the official XACLE dataset download procedure from their GitHub repository.

### 6. Add M2D-CLAP and MGA-CLAP models from GitHub
>1. M2D-CLAP
>```bash
>git clone https://github.com/nttcslab/m2d.git
>```
>Please download the M2D-CLAP's weights `m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025` following the procedure on their repository: [M2D-CLAP](https://github.com/nttcslab/m2d)


>2. MGA-CLAP
>```bash
>git clone https://github.com/Ming-er/MGA-CLAP.git
>```
>Please download the MGA-CLAP's weights following the procedure on their repository: [MGA-CLAP](https://github.com/Ming-er/MGA-CLAP)
>>__Recommended Change:__
>> Comment out the line `from tools.utils import *` --> `#from tools.utils import *` in `MGA-CLAP / models / ase_model.py` \
>> __Reason:__ We are using this model to extract Audio-Text features in inference-only mode, and `tools.utils` file contains packages we don't need for inference, hence I'm preferring to avoid installing those packages. But if you want to use the MGA-CLAP for training feel free to keep `tools.utils`, and yes, you need to install the packages as mentioned in it.
