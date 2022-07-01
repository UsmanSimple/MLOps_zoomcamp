import pickle
import os

from flask import Flask, request, jsonify

import mlflow
from dotenv import load_dotenv
load_dotenv()

TRACKING_URI = os.getenv('server_host')

run_id= '540cc07b5ef34c45aab9fb61837f12e9'


mlflow.set_tracking_uri(f"http://{TRACKING_URI}:5000")

# Load model as a PyFuncModel.
logged_model = f"s3://mlflow-artifact-zoomcamp/1/{run_id}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return preds[0]

app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': run_id
    }

    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
