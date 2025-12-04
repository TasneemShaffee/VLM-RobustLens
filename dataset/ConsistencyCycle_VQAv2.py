import json
from pathlib import Path
from collections import defaultdict

def load_vqa_rephrasings(json_path, images_dir):
    data = json.loads(Path(json_path).read_text())
    rows = []
    
    for q in data["questions"]:
        image_id = q["image_id"]
        group_id = q.get("rephrasing_of", q["question_id"])  
        image_file = Path(images_dir) / f"COCO_{Path(images_dir).name}_{image_id:012d}.jpg"
        rows.append({
            "question_id": q["question_id"],
            "group_id": group_id,
            "image_id": image_id,
            "question": q["question"],
            "valid_answers": q.get("valid_answers"),
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