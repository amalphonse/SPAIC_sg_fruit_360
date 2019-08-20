from flask import Flask, request, render_template

app = Flask(__name__)

from commons import get_tensor
from inference import get_fruit_name


# @app.route('/', methods=['GET', 'POST'])
@app.route('/')
def upload():
    # if request.method == 'GET':
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        print(request.files)
        if 'file' not in request.files:
            print('file not uploaded')
            return
        file = request.files['file']
        category, fruit_name = get_fruit_name(file)
        # get_fruit_name(image_bytes=image)
        # tensor = get_tensor(image_bytes=image)
        # print(get_tensor(image_bytes=image))
        return render_template('result.html', fruit_name=fruit_name, category=category)


if __name__ == '__main__':
    app.run(debug=True)
