# --------------------------------------------------------
# VLM-RobustLens
# Copyright (c) 2025 Brown University
# All rights reserved.
# Licensed under The MIT License [see LICENSE for details]
# Written by Mattew Prenovitz
# --------------------------------------------------------

from pathlib import Path
from collections import defaultdict
import json
from dataset import *

import os
import torch

import argparse
from src import *

from metrics import *
import json
from itertools import islice
import random
from utility import _image_exists, set_text_layers_eager, attach_attention_hooks, detach_hooks, _agg_rows, _group_by_type, _mean_or_nan, _atomic_write_json
import os, json, math, statistics as stats
from collections import defaultdict
def process_batch(groups,runner,skip_vision=False):
    all_rows = []
    for gid, grp in groups.items():
        img = grp["image"]
        qs = grp["questions"]
        if len(qs) < 2: continue
        qA, qB = qs[0], qs[1]

        clear_attn()
        _ = runner.run(img, qA, do_generate=False)
        mA = snapshot()

        clear_attn()
        _ = runner.run(img, qB, do_generate=False)
        mB = snapshot()

        rows = compare_attention_runs(mA, mB, skip_vision=skip_vision)
        for r in rows:
            r["group_id"] = gid
            r["image_id"] = grp["image_id"]
        all_rows.extend(rows)

    with open("attn_robustness.jsonl", "w") as f:
        for r in all_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(all_rows)} per-layer measurements to attn_robustness.jsonl")
def parse_args():
    parser = argparse.ArgumentParser(description="Run attention probe with arguments.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to cache directory.")
    parser.add_argument("--model_name", type=str, required=True, choices=["gemma3", "qwen3vl", "internvl"],help="Model name to load.")
    parser.add_argument("--dataset", type=str, default="cyc", help="Provide dataset name abbreviation: vg or cyc")
    parser.add_argument("--json_path", type=str, default="cyc", help="Provide the filepath to your QA json.")
    parser.add_argument("--image_path", type=str, default="cyc", help="PProvide the filepath to your image directory.")
    parser.add_argument("--save_frequency", type=int, default=5, help="Provide the frequency of saving intermediate results.")
    parser.add_argument(
        "--attn_mode",
        choices=["full", "blocks"],
        default="full",
        help="How to package attention before metric comparison: "
             "'full' = original full matrices, 'blocks' = text/vision blocks (t2t, t2v, v2t, v2v)."
    )
    return parser.parse_args()





def process_dataset(runner, groups, *, model_name, dataset_name, out_json_path,save_frequency=5,max_images=None,sample_mode="first",seed=0,skip_vision=False,attn_mode="full"):
  
    set_text_layers_eager(runner.model, model_name)

    hooks = attach_attention_hooks(runner, model_name)
    try:
        results_payload = {
            "model": model_name,
            "dataset": dataset_name,
            "num_images": 0,
            "per_image": {},     
            "dataset_summary": {},
        }

    
        all_rows_all_images = []
        skipped_missing = 0
        if max_images is not None:
            if sample_mode == "random":
                rng=random.seed(seed)
                keys = list(groups.keys())
                if max_images> len(keys):
                    selected_ids = keys
                else: selected_ids=rng.sample(keys,max_images) 
                iter_items=((gid,groups[gid]) for gid in selected_ids)    
            else:
                iter_items = islice(groups.items(), max_images)
               
        #for gid, grp in groups.items():
        for gid, grp in iter_items:
            img_path   = grp["image"]
            image_id   = grp.get("image_id", gid)
            print(f"Processing image_id={image_id} from group_id={gid}... img_path={img_path}")
            questions  = grp.get("questions", [])
            if not _image_exists(img_path):
                print(f"[SKIP] Missing/corrupted image: {img_path}")
                skipped_missing += 1
                continue
            if not questions:
                continue
            q0 = questions[0]
            paraphrases = questions[1:] if len(questions) > 1 else []

           
            if not paraphrases:
                continue

        
            clear_attn_buffers()
           
            _ = runner.run(img_path, q0, do_generate=False)
            if attn_mode=="blocks":
              maps_A = package_attention_run(vision_attn_weights, text_attn_blocks, mm_token_type_ids=None,attn_mode=attn_mode)
            else:
                maps_A = package_attention_run(vision_attn_weights, text_attn_weights, mm_token_type_ids=None,attn_mode=attn_mode)
            clear_attn_buffers()
            text_attn_blocks.clear()
            vision_attn_weights.clear()
            text_attn_weights.clear()
          
            per_image_pairs = {}  

          
            for idx, qk in enumerate(paraphrases, start=1):
              
                clear_attn_buffers()
                text_attn_blocks.clear()
                vision_attn_weights.clear()
                text_attn_weights.clear()
                _ = runner.run(img_path, qk, do_generate=False)
              
                if attn_mode=="blocks":
                      
             
                      maps_B = package_attention_run(vision_attn_weights, text_attn_blocks, mm_token_type_ids=None,attn_mode=attn_mode)
                else: maps_B = package_attention_run(vision_attn_weights, text_attn_weights, mm_token_type_ids=None,attn_mode=attn_mode)
               
                clear_attn_buffers()
                text_attn_blocks.clear()
                vision_attn_weights.clear()
                text_attn_weights.clear()
                
                rows = compare_attention_runs(maps_A, maps_B,skip_vision=skip_vision) 
               
                per_image_pairs[f"q0|q{idx}"] = rows
                all_rows_all_images.extend(rows)

     
            image_rows = []
            for rows in per_image_pairs.values():
                image_rows.extend(rows)

            per_image_summary = {
                "overall": _agg_rows(image_rows),
                "by_type": _group_by_type(image_rows)
            }

            results_payload["per_image"][str(image_id)] = {
                "group_id": gid,
                "image_path": img_path,
                "num_paraphrases": len(paraphrases),
                "pairs": per_image_pairs,    
                "summary": per_image_summary,
            }
            results_payload["num_images"] += 1
            print("Number of images processed: ",results_payload["num_images"])

            if results_payload["num_images"]%save_frequency==0:
                results_payload["dataset_summary"] = {
                "overall": _agg_rows(all_rows_all_images),
                "by_type": _group_by_type(all_rows_all_images),
                 }
                os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
                _atomic_write_json(results_payload, out_json_path)
                print(f"[CKPT] Wrote partial results to {out_json_path} "
                f"(num_images={results_payload['num_images']})")
            
        results_payload["dataset_summary"] = {
            "overall": _agg_rows(all_rows_all_images),
            "by_type": _group_by_type(all_rows_all_images),
        }
        os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(results_payload, f, indent=2)

        print(f"[OK] wrote results to {out_json_path}")

    finally:
        detach_hooks(hooks)



def main():
    args = parse_args()
    CACHE = args.cache_dir
    dataset_name = "" 
    json_name = args.dataset


    runner = load_runner(args.model_name, cache_dir=CACHE, enable_attn=True)
    print(f"Loaded runner.")
    groups = None

    print("Loading Dataset")
   
    if json_name== "cyc":
            dataset_name = "COCO-rephrase" 
            json_path  = args.json_path
            images_dir = args.image_path
            groups = load_vqa_rephrasings(json_path, images_dir)
    elif json_name== "vg":
            dataset_name = "VG-rephrase"
            json_path = args.json_path
            images_dir = args.image_path
            groups = load_vg_vqa_rephrasings(json_path, images_dir) 

    if groups: 
        print("Done Loading Dataset") 
    else: 
        print("Error Loading Dataset")
        return
    out_json = os.path.join("experiments", dataset_name, args.model_name, "metrics_and_summaries.json")
    process_dataset(runner, groups,
                    model_name=args.model_name,
                    dataset_name=dataset_name,
                    out_json_path=out_json,
                    save_frequency=args.save_frequency,
                    max_images=600,          
                    sample_mode="first",       
                    seed=0,
                    skip_vision=False,
                    attn_mode=args.attn_mode,                    
                    )
if __name__ == "__main__":
    main()