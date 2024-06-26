from flask import Flask, request, render_template
from smile_detector import SmileDetector
from urllib.request import urlopen

app = Flask(__name__)

smile_detector = SmileDetector()

@app.route('/')
def hello_world():  # put application's code here
    return render_template('index.html')

@app.route("/has_smile",  methods = ['POST'])
def has_smile():
    image_data = request.get_data()
    res = smile_detector.smileCheck(image_data)

    if res:
        return "ok"
    return {'smiling': res}

@app.route("/has_smile_json",  methods = ['POST'])
def has_smile_json():
    json_data = request.json
    with urlopen(json_data['image']) as response:
        image_data = response.read()
    foundFaces = smile_detector.findFaces(image_data)
    smiles = smile_detector.smileCheckForFaces(image_data,foundFaces)

    return {'smiling': smiles, 'faces': foundFaces}

if __name__ == '__main__':
    app.run(host='0.0.0.0')
