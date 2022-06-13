import pandas as pd
import requests
import pickle
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


@task
def get_paths(date):
    logger = get_run_logger()

    date_list = sorted([(pd.to_datetime(date)- relativedelta(months=i)).strftime("%Y-%m") for i in range(1, 3)])

    file_paths = []

    for i in date_list:

        file_name = f"fhv_tripdata_{i}"

        result= requests.get(f"https://nyc-tlc.s3.amazonaws.com/trip+data/{file_name}.parquet") 

        file_path = f"Homework/data/{file_name}.parquet"

        with open(file_path, 'wb') as file:

            file.write(result.content)

        logger.info(f'{file_name} has been successfully downloaded and placed in the file path- {file_path}')

        file_paths.append(file_path)

    return file_paths[0], file_paths[1]

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    logger = get_run_logger()

    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow(task_runner=SequentialTaskRunner())
def main(date : str = None):

    logger = get_run_logger()

    if date is None:
        date = datetime.now()
    
    train_path, val_path = get_paths(date).result()


    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()


    with open(f'Homework/models/model-{date}.bin', 'wb') as lin_reg:
        pickle.dump(lr, lin_reg )
    
    dv_path = f'Homework/models/dv-{date}.b'
    
    with open(dv_path, 'wb') as dict_vect:
        pickle.dump(dv, dict_vect)

    logger.info(f"The Dictvectorizer size is -: {os.stat(dv_path).st_size} bytes")

        
    run_model(df_val_processed, categorical, dv, lr)

main(date="2021-08-15")