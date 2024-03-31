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
    return "ko"


@app.route("/find_faces",  methods = ['POST'])
def find_faces():
    image_data = request.get_data()
    res = smile_detector.smileCheck(image_data)

    if res:
        return "ok"
    return "ko"

@app.route("/has_smile_json",  methods = ['POST'])
def has_smile_json():
    json_data = request.json
    with urlopen(json_data['image']) as response:
        image_data = response.read()

    # image_data = requests.get(json_data['image'], stream=True).raw
    res = smile_detector.smileCheck(image_data)

    if res:
        return "ok"
    return "ko"

if __name__ == '__main__':
    app.run()
