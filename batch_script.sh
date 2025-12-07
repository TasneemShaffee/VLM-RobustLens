#!/bin/bash

### job options ###

#SBATCH -J pretrain_procupineVAE_defaults             # name
#SBATCH -N 1 						# all cores are on one node
#SBATCH --cpus-per-task=8                        # number of cpus
#SBATCH -t 24:00:00 			        # time 1 hour per job days	
#SBATCH --mem 100G 				    # memory
#SBATCH --ntasks=1
#SBATCH -p gpu --gres gpu:1
#SBATCH --output=./robustlens_pretrain.out
#SBATCH --error=./robustlens_pretrain.err
module load python


### Replace all yourUsername instances with your cs login ###
### LOAD DATASETS:
#Use this to load to load in VisualGenome images and QA pairs
: <<"COMMENT"
cd /oscar/home/yourUsername/data

mkdir -p RobustVLM/images/vg_100k

wget https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/question_answers.json.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip

unzip question_answers.json.zip -d RobustVLM
unzip images.zip -d RobustVLM/images/vg_100k
unzip images2.zip -d RobustVLM/images/vg_100k

mv RobustVLM/images/vg_100k/VG_100K/* RobustVLM/images/vg_100k/
mv RobustVLM/images/vg_100k/VG_100K_2/* RobustVLM/images/vg_100k/

rmdir RobustVLM/images/vg_100k/VG_100K
rmdir RobustVLM/images/vg_100k/VG_100K_2
rm question_answers.json.zip images.zip images2.zip
cd ../VLM-RobustLens/
COMMENT


#SETUP ROBOLENS VENV:
export SCRATCH="/oscar/scratch/yourUsername"
export VENV_DIR="$SCRATCH/venvs/robolens"
export CACHE_DIR="$VENV_DIR/.cache"
export TMPDIR="$VENV_DIR/.tmp"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

export PIP_CACHE_DIR="$CACHE_DIR/pip"
export XDG_CACHE_HOME="$CACHE_DIR"          
export TORCH_HOME="$CACHE_DIR/torch"        
export HF_HOME="$CACHE_DIR/huggingface"    
export CUDA_CACHE_PATH="$CACHE_DIR/nv"

#generate an access token from huggingface and put it here
export HUGGINGFACE_HUB_TOKEN=your_token

pip install -r requirements.txt
pip install scipy


#If preprocessing, run these two lines
##########RUN THIS BEFORE FULL PIPELINE OTHERWISE YOU WONT HAVE REPHRASINGS FOR VG###########
#cd dataset
# python llm_rephrasing.py

#Run this for full pipeline
python main.py \
--cache_dir "$CACHE_DIR" \
--model_name "qwen3vl" \
--json_path "/oscar/home/yourUsername/data/RobustVLM/preprocessed_question_answers.json" \
--image_path "/oscar/home/yourUsername/data/RobustVLM/images/vg_100k/" \
--dataset "vg" \
--save_frequency 5