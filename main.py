from flask import Flask
from flask.globals import request
import json
from datalabs_chatbot import get_chat_response


app = Flask(__name__)


@app.route('/')
def index():
    return 'Hello'


@app.route('/', methods=['POST'])
def predict():
    data = request.form['chat']
    res = get_chat_response(data)
    response = {
        "res": res,
        "status": "success"
    }
    response_json = json.dumps(response)
    return response_json


if __name__ == "__main__":
    app.run(debug=True)
