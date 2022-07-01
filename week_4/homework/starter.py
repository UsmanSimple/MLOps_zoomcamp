

import pickle
import pandas as pd
import sys

with open('regressor.bin', 'rb') as f_in:
    lr, dv = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename:str):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def prepare_dict(df):
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    return X_val


def apply_model(input_file: str):

    print(f'reading the data from {input_file}...')
    df = read_data(input_file)

    print(f'preparing the features for prediction.......')
    dicts = prepare_dict(df)

    print(f'applying the model...')
    y_pred = lr.predict(dicts)

    print(f'The mean of y_pred is :{y_pred.mean()}')
    return y_pred, df

def save_model_to_s3(output_file, df, y_pred, year, month):
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result["predictions"] = y_pred
    df_result['ride_id'] = df['ride_id']

    # storing the dataframe as parquet file in s3 bucket
    df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
    )

    print(f'save the dataframe to {output_file}...')


def get_paths(taxi_type:str, year : int, month:int):

    input_file = f's3://nyc-tlc/trip data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f's3://nyc-duration-prediction-usman/taxi_type={taxi_type}/year={year:04d}/month={month:02d}.parquet'

    return input_file, output_file


def run():
    taxi_type = sys.argv[1] # 'fhv'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 4

    input_file, output_file = get_paths(taxi_type, year, month)
    prediction, df = apply_model(input_file=input_file)
    save_model_to_s3(
        output_file=output_file,
        df = df,
        y_pred = prediction,
        year = year,
        month=month
    )


if __name__ == '__main__':
    run()








