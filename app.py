from itertools import count
import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template ,url_for ,request,Response
import numpy as np
from pathlib import Path
import database
import prediction
import json
import io
import random
import visualization
from pymongo import MongoClient
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import modelbuild


app = Flask ( __name__ )
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"



# @app.route('/plot1.png')
# def plot_png1():
#     fig = visualization.create_figure2(data1)
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')

# @app.route('/plot2.png')
# def plot_png2():
#     fig = visualization.create_figure2(data2)
#     output = io.BytesIO()
#     FigureCanvas(fig).print_png(output)
#     return Response(output.getvalue(), mimetype='image/png')


def create_figure1(data1):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 8))
    barWidth = 0.25
    normal = data1[0]
    user = data1[1]
    br1 = np.arange(len(normal))
    br2 = [x + barWidth for x in br1]
    ax.bar(br1, normal, color='g', width=barWidth, edgecolor='grey', label='Normal Value')
    ax.bar(br2, user, color='r', width=barWidth, edgecolor='grey', label="Yours Value")
    ax.set_xlabel('Health status defining attributes', fontweight='bold', fontsize=15)
    ax.set_ylabel('Respective values', fontweight='bold', fontsize=15)
    ax.set_xticks([r + barWidth for r in range(len(normal))])
    ax.set_xticklabels(['cp','chol','fbs','exang','oldpeak','slope','ca','thal'])
    ax.legend()
    plt.tight_layout()
    plt.savefig(STATIC_DIR / 'plotng.png')
    plt.close(fig)

def create_figure2(data2):
    plt.close('all')
    fig, ax = plt.subplots(figsize=(12, 8))
    barWidth = 0.25
    normal = data2[0]
    user = data2[1]
    br1 = np.arange(len(normal))
    br2 = [x + barWidth for x in br1]
    ax.bar(br1, normal, color='g', width=barWidth, edgecolor='grey', label='Normal Value')
    ax.bar(br2, user, color='r', width=barWidth, edgecolor='grey', label="Yours Value")
    ax.set_xlabel('Health status defining attributes', fontweight='bold', fontsize=15)
    ax.set_ylabel('Respective values', fontweight='bold', fontsize=15)
    ax.set_xticks([r + barWidth for r in range(len(normal))])
    ax.set_xticklabels(['trestbps','chol','thalach'])
    ax.legend()
    plt.tight_layout()
    plt.savefig(STATIC_DIR / 'plotng2.png')
    plt.close(fig)


def _validate_clinical_ranges(age, trestbps, chol, thalach, oldpeak, ca):
    if not 1 <= age <= 120:
        raise ValueError("Age must be between 1 and 120.")
    if not 80 <= trestbps <= 250:
        raise ValueError("Resting blood pressure must be between 80 and 250 mm Hg.")
    if not 80 <= chol <= 700:
        raise ValueError("Serum cholesterol must be between 80 and 700 mg/dl.")
    if not 40 <= thalach <= 250:
        raise ValueError("Maximum heart rate must be between 40 and 250 bpm.")
    if not 0 <= oldpeak <= 10:
        raise ValueError("ST depression (oldpeak) must be between 0 and 10.")
    if not 0 <= ca <= 3:
        raise ValueError("Major vessels colored must be between 0 and 3.")

@app.route('/')
def home():
    global counter2
    counter2+=1
    return render_template('home.html',all_count=counter2)


global counter
counter=0
global counter2
counter2=0

@app.route('/predict',methods=['POST'])
def predict():
    global data1
    global data2
    global counter
    global counter2
    if request.method  == 'POST':
        try:
            nameofpatient= request.form.get('name', 'Patient')
            age= request.form.get('age', '0')
            sex=request.form.get('sex', 'Male')
            cp= request.form.get('cp', 'Asymptomatic')
            trestbps= request.form.get('trestbps', '0')
            chol= request.form.get('chol', '0')
            fbs= request.form.get('fbs', 'No')
            restecg=request.form.get('restecg', 'Nothing to note')
            thalach=request.form.get('thalach', '0')
            exang=request.form.get('exang', 'No')
            oldpeak=request.form.get('oldpeak', '0')
            slope=request.form.get('slope', 'Flatsloping: minimal change(typical healthy heart)')
            ca=request.form.get('ca', '0')
            thal=request.form.get('thal', 'normal')

            # Validate and normalize numeric fields before model inference.
            age = int(age)
            trestbps = float(trestbps)
            chol = float(chol)
            thalach = float(thalach)
            oldpeak = float(oldpeak)
            ca = int(ca)
            _validate_clinical_ranges(age, trestbps, chol, thalach, oldpeak, ca)

            counter+=1
            if(counter<=50):
                result=prediction.preprocess(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal )
            else:
                #modelbuild.bulidmodel()
                result=prediction.preprocess(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal )
                counter=0
            #database.crudOperation(age,sex,cp,trestbps,restecg,chol,fbs,thalach,exang,oldpeak,slope,ca,thal,result)
            data1,data2=visualization.visualizationpreprocess(age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,result)
            create_figure1(data1)
            create_figure2(data2)
            return render_template ('result.html',prediction = result, nameofpatient=nameofpatient, model_counter=counter, total_counter=counter2)
        except Exception as e:
            print(f"Error: {str(e)}")
            return render_template('error.html', error_message=str(e))

@app.route('/about')
def about():
    return render_template('disease.html')


@app.errorhandler(500)
def internal_error(error):

    return render_template('error.html')


@app.errorhandler(404)
def not_found(error):
    return "404 error",404

if __name__ == '__main__':
    app.run( debug = True)