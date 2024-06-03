import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import sklearn
import pandas as pd

rf_model = pickle.load(open('rf_model.pkl','rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

app = FastAPI()

@app.get('/')
async def root():
    return {'message':'Bienvenido a Movie Advisor!!'}


class Characteristics(BaseModel):
        title: str
        startYear: int
        genres: str
        minutes: int
        personType: str
        personName: str
        personAge: float

@app.post('/predict')
async def predict(characteristics: Characteristics):
    #input_array = []
    title, startYear, genres, minutes, personType, personName, personAge = characteristics.title, characteristics.startYear, characteristics.genres, characteristics.minutes, characteristics.personType, characteristics.personName, characteristics.personAge

    init_df = pd.DataFrame({'titleType':['movie'],'primaryTitle':[title],'startYear':[startYear], 'runtimeMinutes':[minutes],'genres':[genres], 'primaryName':[personName],'category':[personType],'age':[personAge]}
                           #,columns=['titleType','primaryTitle','startYear','runtimeMinutes','genres','primaryName','category','age']
                           )
    #columns_to_be_encoded = pd.DataFrame({'titleType':['movie'],'primaryTitle':[title],'genres':[genres], 'primaryName':[personName],'category':[personType]})
    init_df[['titleType_encoded','primaryTitle_encoded','genres_encoded','primaryName_encoded','category_encoded']] = encoder.transform(init_df[['titleType','primaryTitle','genres','primaryName','category']])
    columns_to_scale=['startYear','runtimeMinutes','age','titleType_encoded','primaryTitle_encoded','genres_encoded','primaryName_encoded','category_encoded']
    init_df[columns_to_scale] = scaler.transform(init_df[columns_to_scale])
    prediction=rf_model.predict(init_df[['startYear','runtimeMinutes','age','titleType_encoded','primaryTitle_encoded','genres_encoded','primaryName_encoded','category_encoded']])
    return {'Predicted Rating': float(prediction)}
    #return init_df.to_dict()
