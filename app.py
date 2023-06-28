import csv
from flask import Flask, render_template,request,redirect,url_for
import diseaseprediction
import io
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

app = Flask(__name__)

temp=[]
temp2=[]
with open('Testing.csv', newline='') as f:
        reader = csv.reader(f)
        symptoms = next(reader)
        symptoms = symptoms[:len(symptoms)-1]
@app.route('/', methods=['GET'])
def dropdown():
        return render_template('includes/default.html', symptoms=symptoms)

@app.route('/default', methods=['GET'])
def dropdown1():
        return render_template('includes/default.html', symptoms=symptoms)

@app.route('/default1', methods=['GET'])
def dropdown2():
        return render_template('includes/default1.html', symptoms=symptoms)
        
@app.route('/disease_predict', methods=['POST'])
def disease_predict():
    temp.clear()
    temp2.clear()
    selected_symptoms = []
    field=request.form.getlist('field')
    for i in field:
     selected_symptoms.append(i)
    print(selected_symptoms)
    if selected_symptoms:
     disease = diseaseprediction.predictDisease(selected_symptoms)
     temp.append(disease["rf_model_prediction"])
     temp.append(disease["decision_tree_prediction"])
     temp.append(disease["knn_prediction"])
     temp.append(disease["final_prediction"])
     for i in selected_symptoms:
      temp2.append(i)
     print(disease)
     return render_template('disease_predict.html',disease=disease["final_prediction"],symptoms=symptoms)
    else:
     return render_template('disease_predict.html',disease=0,symptoms=symptoms)
@app.route('/disease_predict1', methods=['POST'])
def disease_predict1():
    temp.clear()
    temp2.clear()
    selected_symptoms = []
    if(request.form['Symptom1']!="") and (request.form['Symptom1'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom1'])
    if(request.form['Symptom2']!="") and (request.form['Symptom2'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom2'])
    if(request.form['Symptom3']!="") and (request.form['Symptom3'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom3'])
    if(request.form['Symptom4']!="") and (request.form['Symptom4'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom4'])
    if(request.form['Symptom5']!="") and (request.form['Symptom5'] not in selected_symptoms):
        selected_symptoms.append(request.form['Symptom5'])
    if selected_symptoms:
     disease = diseaseprediction.predictDisease(selected_symptoms)
     temp.append(disease["rf_model_prediction"])
     temp.append(disease["decision_tree_prediction"])
     temp.append(disease["knn_prediction"])
     temp.append(disease["final_prediction"])
     for i in selected_symptoms:
      temp2.append(i)
     print(disease)
     return render_template('disease_predict1.html',disease=disease["final_prediction"],symptoms=symptoms)
    else:
     return render_template('disease_predict1.html',disease=0,symptoms=symptoms)
@app.route('/print-plot')
def plot_png():
   fig = diseaseprediction.plot_png()
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')
   
@app.route('/details')
def details():
   dat=diseaseprediction.detail()
   return render_template('details.html',disease=temp,symptoms=temp2,dat=dat)
   
if __name__ == '__main__':
    app.run(debug=True)