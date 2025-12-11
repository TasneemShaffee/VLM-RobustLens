import argparse
import json
import os
import random
from dataset import *
from itertools import islice
from metrics import *
from src import *
from PIL import Image
from utility import (
    set_text_layers_eager,
    attach_attention_hooks,
    detach_hooks,
    _agg_rows,
    _group_by_type,
    _atomic_write_json,
)


def _image_exists(img_path: str) -> bool:
    if not isinstance(img_path, str) or img_path.startswith(("http://", "https://")):
        return True

    if not os.path.isfile(img_path):
        return False

    try:
        with Image.open(img_path) as im:
            im.verify()
        return True
    except Exception:
        return False


def parse_args():
    parser = argparse.ArgumentParser(description="Run attention probe with arguments.")
    parser.add_argument(
        "--cache_dir", type=str, required=True, help="Path to cache directory."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["gemma3", "qwen3vl", "internvl"],
        help="Model name to load.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cyc",
        help="Provide dataset name abbreviation: vg or cyc",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Provide the filepath to your QA json.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Provide the filepath to your image directory.",
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=5,
        help="Provide the frequency of saving intermediate results.",
    )

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
):

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
                rng = random.seed(seed)
                keys = list(groups.keys())
                if max_images > len(keys):
                    selected_ids = keys
                else:
                    selected_ids = rng.sample(keys, max_images)  # type: ignore
                iter_items = ((gid, groups[gid]) for gid in selected_ids)
            else:
                iter_items = islice(groups.items(), max_images)

        print(iter_items[..8])  # type: ignore

        for gid, grp in iter_items:  # type: ignore
            img_path = grp["image"]
            image_id = grp.get("image_id", gid)
            print(
                f"Processing image_id={image_id} from group_id={gid}... img_path={img_path}"
            )
            questions = grp.get("questions", [])
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
            maps_A = package_attention_run(
                vision_attn_weights, text_attn_weights, mm_token_type_ids=None
            )

            clear_attn_buffers()
            per_image_pairs = {}

            for idx, qk in enumerate(paraphrases, start=1):
                clear_attn_buffers()
                _ = runner.run(img_path, qk, do_generate=False)
                maps_B = package_attention_run(
                    vision_attn_weights, text_attn_weights, mm_token_type_ids=None
                )
                clear_attn_buffers()
                rows = compare_attention_runs(maps_A, maps_B, skip_vision=skip_vision)
                per_image_pairs[f"q0|q{idx}"] = rows
                all_rows_all_images.extend(rows)

            image_rows = []
            for rows in per_image_pairs.values():
                image_rows.extend(rows)

            per_image_summary = {
                "overall": _agg_rows(image_rows),
                "by_type": _group_by_type(image_rows),
            }

            results_payload["per_image"][str(image_id)] = {
                "group_id": gid,
                "image_path": img_path,
                "num_paraphrases": len(paraphrases),
                "pairs": per_image_pairs,
                "summary": per_image_summary,
            }
            results_payload["num_images"] += 1

            print("Number of images processed: ", results_payload["num_images"])

            if results_payload["num_images"] % save_frequency == 0:
                results_payload["dataset_summary"] = {
                    "overall": _agg_rows(all_rows_all_images),
                    "by_type": _group_by_type(all_rows_all_images),
                }
                os.makedirs(os.path.dirname(out_json_path) or ".", exist_ok=True)
                _atomic_write_json(results_payload, out_json_path)
                print(
                    f"[CKPT] Wrote partial results to {out_json_path} "
                    f"(num_images={results_payload['num_images']})"
                )

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
    dataset = args.dataset

    runner = load_runner(args.model_name, cache_dir=CACHE, enable_attn=True)
    print(f"Loaded runner.")

    print("Loading Dataset")
    groups = None
    match dataset:
        case "cyc":
            dataset_name = "COCO-rephrase"
            json_path = args.json_path
            images_dir = args.image_path
            groups = load_vqa_rephrasings(json_path, images_dir)
        case "vg":
            dataset_name = "VG-rephrase"
            json_path = args.json_path
            images_dir = args.image_path
            groups = load_vg_vqa_rephrasings(json_path, images_dir)

    if groups:
        print("Done Loading Dataset")
    else:
        print("Error Loading Dataset")
        return

    out_json = os.path.join(
        "experiments", dataset_name, args.model_name, "metrics_and_summaries.json"
    )

    process_dataset(
        runner,
        groups,
        model_name=args.model_name,
        dataset_name=dataset_name,
        out_json_path=out_json,
        save_frequency=args.save_frequency,
        max_images=10000,
        sample_mode="first",
        seed=0,
        skip_vision=False,
    )


if __name__ == "__main__":
    main()
