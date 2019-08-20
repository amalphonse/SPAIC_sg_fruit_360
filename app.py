import torch
from flask import Flask, request, render_template
from inference import get_fruit_name
from PIL import Image
import numpy as np

app = Flask(__name__)


def process(file):
    img = Image.open(file).convert('RGB')
    img = img.resize((224, 224), Image.ANTIALIAS)
    # print(type(img))
    img = np.array(img)
    img = np.broadcast_to(img, (1, 1, 224, 224))
    # print(type(img))
    img_tensor = torch.from_numpy(img)
    # print(img_tensor.shape)
    return img_tensor.float()


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        return render_template('index.html')

    elif request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        tensor = process(file)
        category, fruit_name = get_fruit_name(tensor)
        return render_template('result.html', fruit_name=fruit_name, category=category)


if __name__ == '__main__':
    app.run()
