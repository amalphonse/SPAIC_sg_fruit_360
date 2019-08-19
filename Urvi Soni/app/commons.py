import io
import os
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import torchvision.transforms as transforms

def get_model():
  path = os.path.join('model', 'Inception' + "." + 'pt')
  model = models. densenet121(pretrained=True)
  model.classifier = nn.Linear(2048, 196)
  model.load_state_dict(torch.load(
    path, map_location='cpu'), strict=False)
  model.eval()
  return model

def get_tensor(image_bytes):
  # my_transforms = transforms.Compose([transforms.Resize(256),
  #                       transforms.CenterCrop(224),
  #                       transforms.ToTensor(),
  #                       transforms.Normalize(mean=[0.485, 0.456, 0.406], 
  #                                             std=[0.229, 0.224, 0.225])])

    my_transforms = transforms.Compose([
        transforms.Resize(256),
        #transforms.CenterCrop(224),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
  image = Image.open(io.BytesIO(image_bytes))
  return my_transforms(image).unsqueeze(0)


# if you get error
# in heroku use this function
# just pass fiel
def transform(file):
    img = Image.open(file)
    img = img.resize((180, 180), Image.ANTIALIAS)
    # print(type(img))
    img = np.array(img)
    img = np.broadcast_to(img, (1, 1, 180, 180))
    # print(type(img))
    img_tensor = torch.from_numpy(img)
    # print(img_tensor.shape)
    return img_tensor.float()
