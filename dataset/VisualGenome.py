import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
#use this import or full pipeline
from .llm_rephrasing import back_translate, gram_var_and_syn_rep
#use this import for preprocessing
#from llm_rephrasing import back_translate, gram_var_and_syn_rep


def parse_args():
    parser = argparse.ArgumentParser(description="Run attention probe with arguments.")
    parser.add_argument("--input_json", type=Path, required=True, help="Path to unprocessed VisualGenome JSON")
    parser.add_argument("--output_json", type=Path, required=True, help="Path to save preprocessed VisualGenome JSON")
    return parser.parse_args()

def preprocess_json(json_path, output_path):
    data = json.loads(Path(json_path).read_text())
    flattened_json = [qa for qas in data for qa in qas["qas"]]
    rephrasing_dicts = []
    
    rephrasing_id = max([qa["qa_id"] for qa in flattened_json]) + 1
    
    for qa in flattened_json:
      rephrasings = gram_var_and_syn_rep(qa["question"], count=3) + back_translate(qa["question"], count=3)
      for r in rephrasings:
        rephrasing_dicts.append({
            "qa_id": rephrasing_id,
            "rephrasing_of": qa["qa_id"],
            "group_id": qa["qa_id"],
            "image_id": qa["image_id"],
            "question": r,
            "answer": qa['answer'],
        })
        rephrasing_id += 1
      qas = rephrasing_dicts + flattened_json
      output_json = {"Name": "VisualGenome_with_Rephrasings", "qas": qas}
      try:
        output_path.write_text(json.dumps(output_json, indent=2))
        return
      except Exception as e:
        print(f"Error writing to {output_path}: {e}")






'''
{
  "image": <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=800x600 at 0x7F2F60698610>,
  "image_id": 1,
  "url": "https://cs.stanford.edu/people/rak248/VG_100K_2/1.jpg",
  "width": 800,
  "height": 600,
  "coco_id": null,
  "flickr_id": null,
  "qas": [
    {
      "qa_id": 986768,
      "image_id": 1,
      "question": "What color is the clock?",
      "answer": "Green.",
      "a_objects": [],
      "q_objects": []
    },
    ...
  }
'''
def load_vg_vqa_rephrasings(json_path, images_dir):
    data = json.loads(Path(json_path).read_text())
    rows = []
    i=0
    for qa in data["qas"]:
        image_id = qa["image_id"]
        group_id = qa.get("rephrasing_of", "qa_id")  
        image_file = Path(images_dir) / f"{image_id}.jpg"
        rows.append({
            "question_id": qa["qa_id"],
            "group_id": group_id,
            "image_id": image_id,
            "question": qa["question"],
            "valid_answers": qa['answer'],
            "image": str(image_file),
        })
                
    groups = defaultdict(lambda: {"questions": [], "answers": None, "image_id": None, "image": None})
    for row in rows:
        g = groups[row["group_id"]]
        g["questions"].append(row["question"])
        g["answers"] = g["answers"] or row["valid_answers"]
        g["image_id"] = g["image_id"] or row["image_id"]
        g["image"] = g["image"] or row["image"]
    return groups

def main():
    args = parse_args()

    
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Starting preprocessing of {args.input_json}")
    try:
      preprocess_json(args.input_json, args.output_json)
      print(f"\nPreprocessing finished. Output saved to {args.output_json}")
    except Exception as e:
      print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    main()