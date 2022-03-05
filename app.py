import uvicorn
from fastapi import FastAPI



import json
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

@app.get('/')
def index():
    return {'message': 'Welcome to mlops webinar'}


@app.get('/predict')
def predict(data):
    # data=json.dumps({'sepal length (cm)':1, 'sepal width (cm)':1, 'petal length (cm)':1, 'petal width (cm)':1})
    data=list(map(lambda x: int(x),data.split(',')))
    # print(data)
    prediction = classifier.predict([data])

    return {
        'prediction': prediction.tolist()[0]
    }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)