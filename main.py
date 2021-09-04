from flask import Flask
from flask.globals import request
from flask_cors import CORS
import json
from datalabs_chatbot import get_chat_response


app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})

@app.route('/')
def index():
    return 'Hello'


@app.route('/', methods=['POST'])
def predict():
    data = request.form['chat']
    res = get_chat_response(data)
    response = {
        "res": res["res"],
        "type" : res["type"],
        "tag" : res["tag"],
        "tag2": res["tag2"],
        "course": res["course"],
        "course2": res["course2"],
        "status": "success"
    }
    response_json = json.dumps(response)
    return response_json

@app.route('/chatbot_deteksi_background', methods=['POST'])
def predict_chatbot_deteksi_background():
    data = request.form['chat']
    res = get_chat_response(data)
    response = {
        "res": res["res"],
        "type" : res["type"],
        "tag" : res["tag"],
        "tag2": res["tag2"],
        "course": res["course"],
        "course2": res["course2"],
        "status": "success"
    }
    response_json = json.dumps(response)
    return response_json

@app.route('/chatbot_deteksi_ya_tidak', methods=['POST'])
def predict_chatbot_deteksi_ya_tidak():
    data = request.form['chat']
    res = get_chat_response(data)
    response = {
        "res": res["res"],
        "type" : res["type"],
        "tag" : res["tag"],
        "tag2": res["tag2"],
        "course": res["course"],
        "course2": res["course2"],
        "status": "success"
    }
    response_json = json.dumps(response)
    return response_json

@app.route('/chatbot_deteksi_bahasa', methods=['POST'])
def predict_chatbot_deteksi_bahasa():
    data = request.form['chat']
    res = get_chat_response(data)
    response = {
        "res": res["res"],
        "type" : res["type"],
        "tag" : res["tag"],
        "tag2": res["tag2"],
        "course": res["course"],
        "course2": res["course2"],
        "status": "success"
    }
    response_json = json.dumps(response)
    return response_json

@app.route('/chatbot_deteksi_ya_tidak_karir', methods=['POST'])
def chatbot_deteksi_ya_tidak_karir():
    data = request.form['chat']
    res = get_chat_response(data)
    response = {
        "res": res["res"],
        "type" : res["type"],
        "tag" : res["tag"],
        "tag2": res["tag2"],
        "course": res["course"],
        "course2": res["course2"],
        "status": "success"
    }
    response_json = json.dumps(response)
    return response_json


if __name__ == "__main__":
    app.run(debug=True)
