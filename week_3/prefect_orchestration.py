
import pandas as pd
import numpy as np


from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error

import mlflow
import pickle

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


@task
def read_dataframe(filename):

    df = pd.read_parquet(filename)


    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)


    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

@task
def preprocess_data(train_df, test_df):
    
    train_df['PU_DO'] = train_df['PULocationID'] + '_' + train_df['DOLocationID']
    test_df['PU_DO'] = test_df['PULocationID'] + '_' + test_df['DOLocationID']

    categorical = ['PU_DO'] 
    numerical = ['trip_distance']

    target = 'duration'


    dv = DictVectorizer()


    train_dicts = train_df[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    test_dicts = test_df[categorical + numerical].to_dict(orient='records')
    X_test = dv.transform(test_dicts)

    y_train = train_df[target].values
    y_test = test_df[target].values

    return dv, X_train, y_train, X_test, y_test

@task
def train_model_search(train, valid, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=100,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials()
    )
    return

@task
def train_best_model(train, valid, y_val, dv):
    with mlflow.start_run():

        best_params = {
            'learning_rate': 0.09585355369315604,
            'max_depth': 30,
            'min_child_weight': 1.060597050922164,
            'objective': 'reg:linear',
            'reg_alpha': 0.018060244040060163,
            'reg_lambda': 0.011658731377413597,
            'seed': 42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=100,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

    
@flow(task_runner=SequentialTaskRunner())
def run_main(train_path : str, test_path : str):

    # set the tracking uri for the mlflow- backend with sqlite database
    mlflow.set_tracking_uri("sqlite:///orchestration.db")

    # set experiment name
    mlflow.set_experiment("orchestration-experiment")

    # read the data
    
    train_df = read_dataframe(train_path)
    test_df = read_dataframe(test_path)

    # pre_process data and get features
    dv, X_train, y_train, X_test, y_test = preprocess_data(train_df, test_df).result()
    # get 
    train = xgb.DMatrix(X_train, label=y_train)
    test = xgb.DMatrix(X_test, label=y_test)

    train_model_search(train, test, y_test)
    train_best_model(train, test, y_test, dv)


run_main('./Data/green_tripdata_2021-01.parquet', './Data/green_tripdata_2021-02.parquet')

