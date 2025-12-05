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
>Please download the M2D-CLAP's weights `m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025` following the procedure in their repository: [M2D-CLAP](https://github.com/nttcslab/m2d)


>2. MGA-CLAP
>```bash
>git clone https://github.com/Ming-er/MGA-CLAP.git
>```
>Please download the MGA-CLAP's weights following the procedure in their repository: [MGA-CLAP](https://github.com/Ming-er/MGA-CLAP)
>>__Recommended Change:__
>> Comment out the line `from tools.utils import *` --> `#from tools.utils import *` in `MGA-CLAP / models / ase_model.py` \
>> __Reason:__ We are using this model to extract Audio-Text features in inference-only mode, and `tools.utils` file contains packages we don't need for inference, hence I'm preferring to avoid installing those packages. But if you want to use the MGA-CLAP for training, feel free to keep `tools.utils`. Then, you need to install the packages as mentioned in it.


# Usage
### Training
```
python train.py <config_file>
```
> e.g. `python train.py config_submission1.json`
>> where <config_file> = config_submission1.json

### Inference
```
python inference.py <chkpt_subdir_name> <dataset_key>
```
> e.g. `python inference.py version_config_submission1 validation`
>> where <chkpt_subdir_name> = version_config_submission1 \
>>     <dataset_key> = validation

### Evaluation (Taken from XACLE's official implementation)
```
python evaluate.py <inference_csv_path> <ground_truth_csv_path> <save_results_dir>
```
> e.g. `python evaluate.py outputs/version_config_submission1/inference_result_for_validation.csv datasets/XACLE_dataset/meta_data/validation_average.csv outputs/version_config_submission1/`
>> where <inference_csv_path> = outputs/version_config_submission1/inference_result_for_validation.csv \
>>     <ground_truth_csv_path> = datasets/XACLE_dataset/meta_data/validation_average.csv \
>>     <save_results_dir> = outputs/version_config_submission1/

$${\color{red}This\ is\ red\ text}$$
# Results
|             |            SRCC          |           LCC           |           KTAU          |           MSE           |
|     :-      |            :-:           |           :-:           |           :-:           |           :-:           |
|  Baseline   |           0.384          |          0.396          |          0.264          |          4.386          |
| Submission1 |  $${\color{blue}0.664}$$ | $${\color{blue}0.680}$$ | $${\color{blue}0.483}$$ |          3.114          |
| Submission2 |           0.653          |          0.673          |          0.477          |          3.153          |
| Submission3 |           0.664          |          0.679          |          0.482          | $${\color{blue}3.106}$$ |
| Submission4 |  0.384 | 0.396 | 0.264 | 4.386 |

> Note:
> This repository contains code for `Submission{1,3,4}.` \
> `Submission2` will be developed separately in future by other team member of this project; hyperlink to which will be mentioned here soon.

# Directory Structure
```
EnsembleSVR-XACLE
|___train.py
|___inference.py
|___evaluate.py
|___config_submission1.json
|___config_submission3.json
|___config_submission4.json
|___train_inference_scribble.ipynb

|___outputs
    |___# Your trained model's output will be added in this dir after running train.py

|___load_pretrained_models
    |___load_model.py

|___features
    |___all_feature_dict.py
    |___extract_features.py
    |___proximity_features.py

|___utils
    |___utils.py

|___datasets
    |___XACLE_dataset
        |___wav
            |___train
                |___07407.wav
                |___ . . .
            |___validation
                |___10414.wav
                |___ . . .
            |___test
                |___13499.wav
                |___ . . .
        |___meta_data
            |___train.csv
            |___train_average.csv
            |___validation.csv
            |___validation_average.csv
            |___test.csv
    |___fetch_data.py

|___m2d #(Note: I'm only showing some important files from the m2d repo to arrange the file structure)
    |___m2d_clap_vit_base-80x1001p16x16p16kpBpTI-2025
        |___checkpoint-30.pth
    |___examples
        |___portable_m2d.py

|___MGA-CLAP #(Note: I'm only showing some important files from the MGA-CLAP repo to arrange the file structure)
    |___pretrained_models
        |___mga-clap.pt
    |___models
        |___ase_model.py
    |___settings
        |___inference_example.yaml
```
    

