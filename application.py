from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application= Flask(__name__)

app=application

#Route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods= ['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data= CustomData(
          satisfaction_level=float(request.form.get('satisfaction_level')),
          number_project=request.form.get('number_project'),
          average_montly_hours=request.form.get('average_montly_hours'),
          time_spend_company= request.form.get('time_spend_comapny'),
          department=request.form.get('department'),
          salary=request.form.get('salary')

        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])
    
if __name__=="__main__":
    app.run(host='0.0.0.0')