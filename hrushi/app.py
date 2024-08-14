from flask import Flask, request
import pickle

app = Flask(__name__)
model_pickle = open('hrushi\Scripts\classifier_hrushi.pkl' , 'rb')
clf = pickle.load(model_pickle)

@app.route('/')
def hello_world():
    return 'Hell world'

@app.route('/predictions',methods = ['POST'])

def predictions():
    survival_status=request.get_json()
    Pclass = survival_status['Pclass']
    SibSp = survival_status['SibSp']
    Parch = survival_status['Parch']
    Sex_male = survival_status['Sex_male']
    result = clf.predict([[Pclass,SibSp,Parch,Sex_male]])

    if result ==0:
        pred = "Mela"
   
    else:
        pred = "Jagala"
    return {"survival_status": pred}

    
    







