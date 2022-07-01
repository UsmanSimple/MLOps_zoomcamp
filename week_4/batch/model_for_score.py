
import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

import mlflow 

import os
from dotenv import load_dotenv
load_dotenv()

server_host = os.getenv('server_host')
print(f'server host:{server_host}')
mlflow.set_tracking_uri(f"http://{server_host}:5000")

mlflow.set_experiment('usman-server-scoring-model-experiment')

print(mlflow.list_experiments())

def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df


def prepare_dictionaries(df: pd.DataFrame):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')
    return dicts
    
def prepare_features(df_train, df_val):
    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    dict_train = prepare_dictionaries(df_train)
    dict_val = prepare_dictionaries(df_val)
    return y_train, y_val, dict_train, dict_val

def main(y_train, y_val, dict_train, dict_val):
    with mlflow.start_run():
        params = dict(max_depth= 20, n_estimators=100, min_samples_leaf=10, random_state = 0)
        mlflow.log_params(params)

        pipeline = make_pipeline(DictVectorizer(), RandomForestRegressor(**params, n_jobs= -1))

        pipeline.fit(dict_train, y_train)

        y_pred = pipeline.predict(dict_val)

        rmse = mean_squared_error(y_pred, y_val, squared=False)
        print(params, rmse)

        mlflow.log_metric('rmse', rmse)

        mlflow.sklearn.log_model(pipeline, artifact_path = 'model')
    
def run():
    print('reading the train_set>>>>>>>>>>>>>>>>>>>>>>')
    df_train = read_dataframe('../../week_2/Data/green_tripdata_2021-01.parquet')
    print('reading the test_set>>>>>>>>>>>>>>>>>>>>>>')
    df_val = read_dataframe('../../week_2/Data/green_tripdata_2021-02.parquet')
    print('preparing the features ........................>>>>>>>>>>>>>>>>>>>>>>')
    y_train, y_val, dict_train, dict_val = prepare_features(df_train, df_val)
    print('running the mlflow_model..................................')
    main(y_train, y_val, dict_train, dict_val)


if __name__ == '__main__':
    run()