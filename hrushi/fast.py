from fastapi import FastAPI, Request
from pydantic import BaseModel
import pickle

app = FastAPI()

# Load the classifier model
with open('hrushi/Scripts/classifier_hrushi.pkl', 'rb') as model_pickle:
    clf = pickle.load(model_pickle)

# Define a Pydantic model for the request body
class SurvivalStatus(BaseModel):
    Pclass: int
    SibSp: int
    Parch: int
    Sex_male: int

@app.get("/")
async def hello_world():
    return {"message": "Hello World"}

@app.post("/predictions")
async def predictions(survival_status: SurvivalStatus):
    data = [
        survival_status.Pclass,
        survival_status.SibSp,
        survival_status.Parch,
        survival_status.Sex_male
    ]
    result = clf.predict([data])

    if result[0] == 0:
        pred = "Mela"
    else:
        pred = "Jagala"
    return {"survival_status": pred}
