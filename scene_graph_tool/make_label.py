import json
import sys
from pathlib import Path


def names_by_id(items):
  """Tạo list 1-indexed, giữ index khớp với output của model."""
  max_id = max(item["id"] for item in items)
  names = ["__background__"] + ["__unknown__"] * max_id

  for item in items:
      names[item["id"]] = item["name"]

  return names


categories_path = Path("/home/whopqa/Documents/Research/VLN/scenegraph/SGG-Benchmark/docs/categories.json")
output_path = Path("/home/whopqa/Documents/Research/VLN/scenegraph/SGG-Benchmark/scene_graph_tool/psg_labels.json")

with categories_path.open(encoding="utf-8") as f:
  categories = json.load(f)

object_names = names_by_id(categories["categories"])
predicate_names = names_by_id(categories["rel_categories"])

output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
  json.dump(
      {
          "objects": object_names,
          "predicates": predicate_names,
      },
      f,
      indent=2,
      ensure_ascii=False,
  )

print(f"Object classes: {len(object_names) - 1}")
print(f"Predicate classes: {len(predicate_names) - 1}")
print(f"Saved labels: {output_path}")
