from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
import pandas as pd
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
CORS(app)

# Activation Functions
def logistic(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh_activation(x):
    return np.tanh(x)

@app.route('/forward', methods=['POST'])
def forward():
    data = request.json
    input_data = np.array(data['input'])
    weights = [np.array(w) for w in data['weights']]
    biases = [np.array(b) for b in data['biases']]
    activation_func = data['activation']

    if activation_func == "logistic":
        func = logistic
    elif activation_func == "relu":
        func = relu
    elif activation_func == "tanh":
        func = tanh_activation

    for w, b in zip(weights, biases):
        input_data = func(np.dot(input_data, w) + b)

    return jsonify({"output": input_data.tolist()})

@app.route('/saveData', methods=['POST'])
def saveData(): 
    file = request.files['file']
    df = pd.read_csv(file)
    normalize = request.form.get('normalize') == 'true'
    skipHeader = request.form.get('skipHeader') == 'true'
    validationSplit = float(request.form.get('validationSplit'))
    print(normalize, skipHeader, validationSplit)

    return jsonify({"status": "success"})

if __name__ == '__main__':
    print("Starting server...")
    app.run(port=5000)
