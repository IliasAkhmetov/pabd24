"""House price prediction service"""
import os
from dotenv import dotenv_values
from flask import Flask, request, url_for
from flask_cors import CORS
from joblib import load
from flask_httpauth import HTTPTokenAuth
from flask import send_from_directory
from utils import predict_io_bounded, predict_cpu_bounded, predict_cpu_multithread

MODEL_SAVE_PATH = 'models/catboost_regression_v01.joblib'

app = Flask(__name__)
CORS(app)

config = dotenv_values(".env")
auth = HTTPTokenAuth(scheme='Bearer')

tokens = {
    config['APP_TOKEN']: "user1",
}

model = load(MODEL_SAVE_PATH)


@auth.verify_token
def verify_token(token):
 if token in tokens:
     return tokens[token]

#
#     area = float(in_data['area'])
#     floor = int(in_data['floor'])
#     floors_count = int(in_data['floors_count'])
#     is_first = (floor == 1)
#     is_last = (floor == floors_count)
#     price = model.predict([[area,
#                             is_first,
#                             is_last,
#                             floors_count]])
#     return int(price)

def predict(in_data: dict) -> int:
    """ Predict house price from input data parameters.
    :param in_data: house parameters.
    :raise Error: If something goes wrong.
    :return: House price, RUB.
    :rtype: int
    """
    total_meters = float(in_data['area'])
    floor = int(in_data['floor'])
    floors_count = int(in_data['floors_count'])
    first_floor = int(in_data['first_floor'])
    last_floor = int(in_data['last_floor'])
    underground = in_data['underground']

    price = model.predict([[total_meters, first_floor, last_floor, floors_count, underground]])
    return int(price)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route("/")
def home():
    return """
    <html>
    <head>
    <link rel="shortcut icon" href="/favicon.ico">
    </head>
    <body>
    <h1>Housing price service.</h1> Use /predict endpoint
    </body>
    </html>
    """


@app.route("/predict", methods=['POST'])
@auth.login_required
def predict_web_serve():
    """Dummy service"""
    in_data = request.get_json()
    price = predict(in_data)
    return {'price': price}


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
