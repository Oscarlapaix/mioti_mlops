print('Hello Word')
import pickle
from fastapi import FastAPI

app=FastAPI()

@app.get('/')
async def root():
    return {'message':'Bienvenido a Movie Advisor'}
