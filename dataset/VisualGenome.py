import json
from pathlib import Path
from collections import defaultdict
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
    for qa in data["qas"]:
        image_id = qa["image_id"]
        group_id = qa.get("rephrasing_of", qa["qa_id"])  
        image_file = Path(images_dir) / f"COCO_{Path(images_dir).name}_{image_id:012d}.jpg"
        rows.append({
            "question_id": qa["qa_id"],
            "group_id": group_id,
            "image_id": image_id,
            "question": qa["question"],
            "valid_answers": qa['answer'],
            "image": str(image_file),
        })
        if group_id == qa["qa_id"]:
            rephrasings = gram_var_and_syn_rep(qa["question"], count=3) + back_translate(qa["question"], count=3)
            for r in rephrasings:
                rows.append({
                     "rephrasing_of": qa["qa_id"],
                      "group_id": group_id,
                      "image_id": image_id,
                      "question": r,
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