from flask import Flask, request
from smile_detector import SmileDetector
from io import BytesIO
app = Flask(__name__)

smile_detector = SmileDetector()

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'

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

if __name__ == '__main__':
    app.run()
