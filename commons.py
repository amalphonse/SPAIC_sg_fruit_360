import io
import os
import torch
import torch.nn as nn
from torchvision import models


def get_model():
    path = os.path.join('Urvi Soni/app/model', 'Inception' + "." + 'pt')
    model = models.inception_v3(pretrained=True)
    model.classifier = nn.Linear(2048, 114)
    model.load_state_dict(torch.load(
        path, map_location='cpu'), strict=False)
    model.eval()
    return model
