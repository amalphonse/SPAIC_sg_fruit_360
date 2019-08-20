import json
import torch

from commons import get_model, get_tensor, transform


with open('Classes_clean.json') as f:
    class_to_name = json.load(f)

# with open('class_to_idx.json') as f:
#   class_to_idx = json.load(f)


idx_to_class = {v: k for k, v in class_to_name.items()}

model = get_model()


def get_fruit_name(file):
    tensor = transform(file)
    outputs = model.forward(tensor)
    _, prediction = outputs.max(1)
    category = prediction.item()
    print(category)
    class_idx = idx_to_class[category.item()]
    fruit_name = class_to_name[class_idx]
    return category, fruit_name
