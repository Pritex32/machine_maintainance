import pandas as pd
import numpy as np
import joblib
import streamlit as st

df=pd.read_csv('C:\\Users\\USER\\Documents\\dataset\\predictive_maintenance_dataset.csv')

df.head()

df.isnull().sum()
df.duplicated().sum()

dfr=df.drop_duplicates(inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['device']=le.fit_transform(df['device'])

x=df[[ 'device', 'metric1', 'metric2', 'metric3', 'metric4',
       'metric5', 'metric6', 'metric7',  'metric9']]
y=df['failure']

from imblearn .over_sampling import SMOTE
smote=SMOTE()
x_resampled,y_resampled=smote.fit_resample(x,y)

y_resampled.value_counts()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

scaler=MinMaxScaler(feature_range=(0,1))

x_train,x_test,y_train,y_test=train_test_split(x_resampled,y_resampled,test_size=0.3,random_state=40)

x_train_v=scaler.fit_transform(x_train)
x_test_v=scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(ccp_alpha=0.002,n_estimators=100)
rf.fit(x_train_v,y_train)

ypred=rf.predict(x_test_v)

from sklearn import metrics
from sklearn.metrics import accuracy_score

accuracy_score(y_test,ypred)
model=joblib.dump(rf,'machine maintainnance_model.joblib')

# fastapi app
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app=FastAPI()

class machinepred(BaseModel):
     device:str
     metric1:float
     metric2:float
     metric3:float
     metric4:float
     metric5:float
     metric6:float
     metric7:float
     metric9:float


@app.get('/')
def home():
     return{'machine maintainance predictions'}

@app.post('/predict')
def predict(request:machinepred):
    data=le.fit_transform([request.device, request.metric1, request.metric2, request.metric3,request.metric4,
                           request. metric5,request. metric6,request. metric7, request. metric9])
    predictions=rf.predict(scaler.transform([data]))
    pred_map={0:'non_failure',
               1:'failure'}
    return{'predictions':pred_map[int(predictions[0])]}

if __name__ == "__main__":
     uvicorn.run(app)
