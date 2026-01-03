# VLM-RobustLens: Analyzing Visual Grounding Robustness of Vision–Language Models under Question Paraphrasing

This work investigates the robustness of Vision–Language Models (VLMs) against diverse types of question paraphrasing across multiple Visual Question Answering (VQA) datasets.
We focus on quantifying attention map drift to reveal how paraphrasing alters a model’s visual grounding and reasoning behavior. Through systematic analysis, we aim to identify which components of VLMs are most sensitive to linguistic variation and which types of paraphrases cause the largest performance and attention shifts. The findings offer actionable insights and design guidelines for developing future VLMs that maintain stable visual grounding and robust understanding under natural language rewordings.


## Setup Instructions

### 1. Create and Activate a Virtual Environment
It’s recommended to use a virtual environment to keep dependencies isolated.

```bash
# Create a virtual environment
python3 -m venv robolens

# Activate the environment
# On Linux/Mac:
source robolens/bin/activate

```
### 2. Install dependencies 
```bash
pip install -r requirements.txt

```
 or run

 ```bash
./setup.bash

```

## Datasets Preparation:
1. From the current parent directory, write the following commands to download the VQAv2 validation: 
```bash
mkdir Datasets
cd Datasets
wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
```
2. The VQA-Rephrasings dataset can be found below:

https://facebookresearch.github.io/VQA-Rephrasings/

It should be saved under Datasets to have structure as follows:

- Datasets
  - compressed
    - v2_mscoco_valrep2014_humans_og_annotations.json
    - v2_OpenEnded_mscoco_valrep2014_humans_og_questions.json
## Inference


1. To run the full pipeline on VQA-Paraphrasing dataset (human paraphrased dataset):

```bash
python main.py --cache_dir <cashe directory that stores huggingface models> --model_name "internvl" --save_frequency 5
```
The choices for --model_name configuration: "internvl", "gemma3", "qwen3vl"

2. To run the stress analysis pipeline on VQA-Paraphrasing dataset :

```bash
python  main_stress_analysis.py --cache_dir <cashe directory that stores huggingface models> --model_name "qwen3vl" --save_frequency 5 --attn_mode blocks
```
The choices for --model_name configuration: "internvl", "gemma3", "qwen3vl"

3. To evaluate the generated answers of the VLM models for the stress analysis:

```bash
python  evaluate_results_by_judge.py --cache_dir <cashe directory that stores huggingface models> --model_name "qwen3vl" --save_frequency 5
```
The choices for --model_name configuration: "internvl", "gemma3", "qwen3vl"


