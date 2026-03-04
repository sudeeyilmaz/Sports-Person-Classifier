from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import util

app = Flask(__name__, static_folder="../UI", template_folder="../UI",static_url_path='')
CORS(app)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify_image', methods=['GET', 'POST'])
def classify_image():
    image_data = request.form['image_data']

    response = jsonify(util.classify_image(image_data))
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response

if __name__ == "__main__":
    print("Starting Python Flask Server For Sports Celebrity Image Classification")
    util.load_saved_artifacts()
    app.run(port=5000)