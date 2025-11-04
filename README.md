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

### 3. Inference 

To run the inference of vlm using huggingface api on one image:  

```bash
cd src

python vlm_inference.py --cache_dir <directory to save the vlm> --model_name "qwen3vl" 
```
To run the inference of vlm using huggingface api on one image  and check the accessible attention maps:

```bash
python vlm_inference.py --cache_dir <directory to save the vlm> --model_name "qwen3vl" --enable_attn_checker
```
