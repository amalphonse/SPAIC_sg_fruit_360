import json

from commons import get_model

with open('Classes_clean.json') as f:
    class_to_name = json.load(f)

idx_to_class = {v: k for k, v in class_to_name.items()}

model = get_model()


def get_fruit_name(tensor):
    outputs = model.forward(tensor)
    _, prediction = outputs.max(1)
    category = prediction.item()
    class_idx = idx_to_class[category]
    fruit_name = class_to_name[class_idx]
    return category, fruit_name
