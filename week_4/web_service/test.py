import requests


ride = {
    "PULocationID": 10,
    "DOLocationID": 50,
    "trip_distance": 40
}


url = 'http://localhost:8080/predict'
response = requests.post(url, json=ride, timeout=10)
print(response.json())