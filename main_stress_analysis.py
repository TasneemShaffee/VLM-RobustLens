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
from stress_questions import STRESS_BANK
from stress_helper import *
import gc, torch

def hard_clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
def _to_serializable(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().cpu().tolist()
    return x

def _select_layers(num_layers: int, spec: str):
    spec = str(spec).strip().lower()
    if spec == "all":
        return list(range(num_layers))
    if "," in spec:
        out = []
        for s in spec.split(","):
            s = s.strip()
            if not s:
                continue
            out.append(int(s))
      
        out2 = []
        for li in out:
            if li < 0:
                li = num_layers + li
            out2.append(max(0, min(num_layers - 1, li)))
        return sorted(set(out2))
  
    li = int(spec)
    if li < 0:
        li = num_layers + li
    return [max(0, min(num_layers - 1, li))]

def collect_text_blocks_per_layer(text_attn_blocks, *, avg_heads=True, layer_spec="-1"):
   
    num_layers = len(text_attn_blocks)
    keep_layers = _select_layers(num_layers, layer_spec)

    layers = {}
    for li in keep_layers:
        blk = text_attn_blocks[li]
        if not isinstance(blk, dict):
            continue

        layer_out = {}
        
        for key in ("t2t", "t2v", "v2t", "v2v"):
            mat = blk.get(key, None)
            if mat is None:
                layer_out[key] = None
                continue

            
            mat = _maybe_avg_heads(mat, avg=avg_heads)  

           

            layer_out[key] = mat

        layers[str(li)] = layer_out

    return layers
def parse_args():
    parser = argparse.ArgumentParser(description="Run attention probe with arguments.")
    parser.add_argument("--cache_dir", type=str, required=True, help="Path to cache directory.")
    parser.add_argument("--model_name", type=str, required=True, choices=["gemma3", "qwen3vl", "internvl"],help="Model name to load.")
    parser.add_argument("--save_frequency", type=int, default=5, help="Provide the frequency of saving intermediate results.")
    parser.add_argument(
        "--attn_mode",
        choices=["full", "blocks"],
        default="full",
        help="How to package attention before metric comparison: "
             "'full' = original full matrices, 'blocks' = text/vision blocks (t2t, t2v, v2t, v2v)."
    )
    parser.add_argument("--save_attn", action="store_true", help="Save per-image per-variant attention maps to JSON.")
    parser.add_argument("--save_heads", action="store_true", help="If set, do NOT average over heads; store head dimension.")
    parser.add_argument("--attn_out", type=str, default="attn_dump.json", help="Where to save attention maps JSON.")
    parser.add_argument("--attn_layers", type=str, default="-1", help="Which layers to save: '-1' or 'all' or '0,5,10'.")
    return parser.parse_args()



def process_dataset(
    runner,
    groups,
    *,
    model_name,
    dataset_name,
    out_json_path,
    save_frequency=5,
    max_images=None,
    sample_mode="first",
    seed=0,
    skip_vision=False,
    attn_mode="blocks",  
    answers_json_path="answers.json",
    do_generate=False,

   
    save_attn=True,
    save_heads=False,             
    attn_out_path="attn_dump.json",
    attn_layers_spec="-1",        
):
   
  
    set_text_layers_eager(runner.model, model_name)

    hooks = attach_attention_hooks(runner, model_name)

   
    avg_heads = (not save_heads)

  
    results_payload = {
        "model": model_name,
        "dataset": dataset_name,
        "num_images": 0,
        "per_image": {},
        "dataset_summary": {},
    }

    answers_payload = {
        "model": model_name,
        "dataset": dataset_name,
        "num_images": 0,
        "per_image": {} 
    }

    attn_payload = {
        "model": model_name,
        "dataset": dataset_name,
        "attn_mode": attn_mode,
        "avg_heads": avg_heads,
        "save_heads": save_heads,
        "layers_spec": attn_layers_spec,
        "num_images": 0,
        "per_image": {}  
    }

    all_rows_all_images = []
    skipped_missing = 0


    if max_images is not None:
        if sample_mode == "random":
            rng = random.Random(seed)
            keys = list(groups.keys())
            selected_ids = keys if max_images >= len(keys) else rng.sample(keys, max_images)
            iter_items = ((gid, groups[gid]) for gid in selected_ids)
        else:
            iter_items = islice(groups.items(), max_images)
    else:
        iter_items = groups.items()

    try:
        for gid, grp in iter_items:
            img_path = grp["image"]
            image_id = grp.get("image_id", gid)
            print(f"Processing image_id={image_id} from group_id={gid}... img_path={img_path}")

            if not _image_exists(img_path):
                print(f"[SKIP] Missing/corrupted image: {img_path}")
                skipped_missing += 1
                continue

          
            q0, variants = build_variants_for_gid(STRESS_BANK, gid)
            print("grp:", grp)
            if q0 is None or not variants:
                print(f"[SKIP] No variants found for group_id={gid}")
                continue

           
            clear_attn_buffers()
            text_attn_blocks.clear()
            vision_attn_weights.clear()
            text_attn_weights.clear()
            hard_clear()
            if model_name == "internvl":
                if do_generate:
                    _ = runner.run(img_path, q0)  
                    ans1 = runner.generate_response(img_path, q0, do_generate=do_generate, gen_kwargs={"max_new_tokens": 2096})
                else:
                    ans1 = runner.run(img_path, q0, do_generate=do_generate, gen_kwargs={"max_new_tokens": 2096})
            else:
                ans1 = runner.run(img_path, q0, do_generate=do_generate, gen_kwargs={"max_new_tokens": 2096})

         
           
            answers_rec = {
                "group_id": gid,
                "image_id": image_id,
                "image_path": img_path,
                "q0": {"question": q0, "answer": ans1} if do_generate else  {"question": q0, "answer": None},
                "original_answer": grp.get("valid_answers"),
                "variants": []
            }
          
           
            if save_attn:
                if attn_mode != "blocks":
                    raise ValueError("To save t2t/t2v/v2t/v2v, set attn_mode='blocks'.")

                q0_layers = collect_text_blocks_per_layer(
                    text_attn_blocks,
                    avg_heads=avg_heads,
                    layer_spec=attn_layers_spec
                )
                # serialize
                q0_layers_ser = {
                    li: {k: _to_serializable(v) for k, v in mats.items()}
                    for li, mats in q0_layers.items()
                }

                attn_rec = {
                    "group_id": gid,
                    "image_id": image_id,
                    "image_path": img_path,
                    "q0": {"type": "original", "text": q0, "layers": q0_layers_ser},
                    "variants": []
                }
            else:
                attn_rec = None

            
            if attn_mode == "blocks":
                maps_A = package_attention_run(
                    vision_attn_weights,
                    text_attn_blocks,
                    mm_token_type_ids=None,
                    avg_heads=avg_heads,        
                    attn_mode=attn_mode
                )
                #print("len(text_attn_blocks) =", len(text_attn_blocks))
                #print("keys in first layer:", text_attn_blocks[0].keys())
                #print("t2v is None?", text_attn_blocks[0].get("t2v") is None)
            else:
                maps_A = package_attention_run(
                    vision_attn_weights,
                    text_attn_weights,
                    mm_token_type_ids=None,
                    avg_heads=avg_heads,
                    attn_mode=attn_mode
                )

        
            clear_attn_buffers()
            text_attn_blocks.clear()
            vision_attn_weights.clear()
            text_attn_weights.clear()
            hard_clear()
           
            per_image_pairs = {}

            for idx, item in enumerate(variants, start=1):
                qk_type = item["type"]
                qk_text = item["text"]

                clear_attn_buffers()
                text_attn_blocks.clear()
                vision_attn_weights.clear()
                text_attn_weights.clear()

                if model_name == "internvl" and do_generate:
                     
                    _ = runner.run(img_path, qk_text)
                    ans2 = runner.generate_response(img_path, qk_text, do_generate=do_generate, gen_kwargs={"max_new_tokens": 2096})
                else:
                    ans2 = runner.run(img_path, qk_text, do_generate=do_generate, gen_kwargs={"max_new_tokens": 2096})

              

                answers_rec["variants"].append({
                    "idx": idx,
                    "type": qk_type,
                    "question": qk_text,
                    "answer": ans2 if do_generate else None
                })

             
                if save_attn:
                    qk_layers = collect_text_blocks_per_layer(
                        text_attn_blocks,
                        avg_heads=avg_heads,
                        layer_spec=attn_layers_spec
                    )
                    qk_layers_ser = {
                        li: {k: _to_serializable(v) for k, v in mats.items()}
                        for li, mats in qk_layers.items()
                    }
                    attn_rec["variants"].append({
                        "idx": idx,
                        "type": qk_type,
                        "text": qk_text,
                        "layers": qk_layers_ser
                    })

              
                if attn_mode == "blocks":
                    maps_B = package_attention_run(
                        vision_attn_weights,
                        text_attn_blocks,
                        mm_token_type_ids=None,
                        avg_heads=avg_heads,
                        attn_mode=attn_mode
                    )
                else:
                    maps_B = package_attention_run(
                        vision_attn_weights,
                        text_attn_weights,
                        mm_token_type_ids=None,
                        avg_heads=avg_heads,
                        attn_mode=attn_mode
                    )

               
                clear_attn_buffers()
                text_attn_blocks.clear()
                vision_attn_weights.clear()
                text_attn_weights.clear()

                rows = compare_attention_runs(maps_A, maps_B, skip_vision=skip_vision)

              
                for r in rows:
                    r["q_variant_type"] = qk_type
                    r["q_variant_text"] = qk_text
                    r["q0_text"] = q0

                per_image_pairs[f"q0|q{idx}"] = rows
                all_rows_all_images.extend(rows)

          
            answers_payload["per_image"][str(image_id)] = answers_rec
            answers_payload["num_images"] += 1

            if save_attn:
                attn_payload["per_image"][str(image_id)] = attn_rec
                attn_payload["num_images"] += 1

           
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
                "num_paraphrases": len(variants),
                "pairs": per_image_pairs,
                "summary": per_image_summary,
            }
            results_payload["num_images"] += 1

            print("Number of images processed:", results_payload["num_images"])

           
            if results_payload["num_images"] % save_frequency == 0:
                results_payload["dataset_summary"] = {
                    "overall": _agg_rows(all_rows_all_images),
                    "by_type": _group_by_type(all_rows_all_images),
                }

                os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
                _atomic_write_json(results_payload, out_json_path)
                print(f"[CKPT] Wrote partial results to {out_json_path} (num_images={results_payload['num_images']})")
                if do_generate:
                    os.makedirs(os.path.dirname(answers_json_path) or ".", exist_ok=True)
                    _atomic_write_json(answers_payload, answers_json_path)
                    print(f"[CKPT] Wrote answers to {answers_json_path} (num_images={answers_payload['num_images']})")

                if save_attn:
                    os.makedirs(os.path.dirname(attn_out_path) or ".", exist_ok=True)
                    _atomic_write_json(attn_payload, attn_out_path)
                    print(f"[CKPT] Wrote attn dump to {attn_out_path} (num_images={attn_payload['num_images']})")

     
        results_payload["dataset_summary"] = {
            "overall": _agg_rows(all_rows_all_images),
            "by_type": _group_by_type(all_rows_all_images),
        }

        os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
        with open(out_json_path, "w") as f:
            json.dump(results_payload, f, indent=2)
        print(f"[OK] wrote results to {out_json_path}")

        os.makedirs(os.path.dirname(answers_json_path) or ".", exist_ok=True)
        with open(answers_json_path, "w") as f:
            json.dump(answers_payload, f, indent=2)
        print(f"[OK] wrote answers to {answers_json_path}")

        if save_attn:
            os.makedirs(os.path.dirname(attn_out_path) or ".", exist_ok=True)
            with open(attn_out_path, "w") as f:
                json.dump(attn_payload, f, indent=2)
            print(f"[OK] wrote attention dump to {attn_out_path}")

    finally:
        detach_hooks(hooks)



def main():
    args = parse_args()
    CACHE = args.cache_dir
    dataset_name = "COCO-rephrase"  

    runner = load_runner(args.model_name, cache_dir=CACHE, enable_attn=True)

 
    json_path  = "./Datasets/compressed/v2_OpenEnded_mscoco_valrep2014_humans_og_questions.json"
    images_dir = "./Datasets/val2014/val2014/"
    print("Loading Dataset")
    groups = load_vqa_rephrasings(json_path, images_dir)
    print("Done Loading Dataset")

    out_json = os.path.join("experiments", dataset_name, args.model_name, "metrics_and_summaries_attent_type.json")
    answers_json_path = os.path.join("experiments", dataset_name, args.model_name, "answers_type.json")
    attn_out_path= os.path.join("experiments", dataset_name, args.model_name, "attn_dump.json")
    process_dataset(runner, groups,
                    model_name=args.model_name,
                    dataset_name=dataset_name,
                    out_json_path=out_json,
                    save_frequency=args.save_frequency,
                    max_images=10,          
                    sample_mode="first",      
                    seed=0,
                    skip_vision=True,
                    attn_mode=args.attn_mode,  
                    answers_json_path=answers_json_path,   
                    do_generate=False,
                    save_attn=True,
                    save_heads=False,                
                    attn_out_path=attn_out_path,#"attn_dump.json",
                    attn_layers_spec="-1"                 
                    )
if __name__ == "__main__":
    main()
