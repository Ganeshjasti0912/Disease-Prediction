import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np


rows, cols = (4, 2)
arr = [[0]*cols]*rows
fig = Figure(figsize=(18, 40))
#importing Training.csv file removing missing values
data = pd.read_csv(os.path.join("templates", "Training.csv")).dropna(axis = 1)
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})

axis = fig.add_subplot(5, 1, 1)
sns.barplot(x = "Disease", y = "Counts", data = temp_df,ax=axis)
axis.set_xticklabels(temp_df["Disease"], rotation = 90)

encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
  X, y, test_size = 0.2, random_state = 24)
 
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")

#Training and testing KNN Classifier
knn = KNeighborsClassifier(n_neighbors=15)
  
knn.fit(X_train, y_train)

preds=knn.predict(X_test)

temp=accuracy_score(y_train, knn.predict(X_train))
arr[0][0]=temp
print(f"Accuracy on train data by KNN Classifier\
: {temp*100}")

temp=accuracy_score(y_test, preds)
arr[0][1]=temp
print(f"Accuracy on test data by KNN Classifier\
: {temp*100}")
cf_matrix = confusion_matrix(y_test, preds)
axis1 = fig.add_subplot(5, 1, 2)
sns.heatmap(cf_matrix, annot=True,ax=axis1)
axis1.set_title("Confusion Matrix for KNN Classifier on Test Data")
 
#Training and testing DecisionTree Classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
preds = dt.predict(X_test)
temp=accuracy_score(y_train, dt.predict(X_train))
arr[1][0]=temp
print(f"Accuracy on train data by Decision Tree Classifier\
: {temp*100}")

temp=accuracy_score(y_test, preds)
arr[1][1]=temp
print(f"Accuracy on test data by Decision Tree Classifier\
: {temp*100}")
cf_matrix = confusion_matrix(y_test, preds)
axis2 = fig.add_subplot(5, 1, 3)
sns.heatmap(cf_matrix, annot=True,ax=axis2)
axis2.set_title("Confusion Matrix for Decision Tree Classifier on Test Data")
 
#Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
temp=accuracy_score(y_train, rf_model.predict(X_train))
arr[2][0]=temp
print(f"Accuracy on train data by Random Forest Classifier\
: {temp*100}")

temp=accuracy_score(y_test, preds)
arr[2][1]=temp
print(f"Accuracy on test data by Random Forest Classifier\
: {temp*100}")
 
cf_matrix = confusion_matrix(y_test, preds)
axis3 = fig.add_subplot(5, 1, 4)
sns.heatmap(cf_matrix, annot=True,ax=axis3)
axis3.set_title("Confusion Matrix for Random Forest Classifier on Test Data")

final_knn = KNeighborsClassifier(n_neighbors=15)
final_dt = DecisionTreeClassifier()
final_rf_model = RandomForestClassifier(random_state=18)
final_knn.fit(X, y)
final_dt.fit(X, y)
final_rf_model.fit(X, y)
 
#Reading the test data
test_data = pd.read_csv("./templates/Testing.csv").dropna(axis=1)
 
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
 
#Making prediction by take mode of predictions
#made by all the classifiers
knn_preds = final_knn.predict(test_X)
dt_preds = final_dt.predict(test_X)
rf_preds = final_rf_model.predict(test_X)
 
final_preds = [mode([i,j,k])[0][0] for i,j,
               k in zip(knn_preds, dt_preds, rf_preds)]
 
temp=accuracy_score(test_Y, final_preds)
arr[3][0]=temp
print(f"Accuracy on Test dataset by the combined model\
: {temp*100}")
 
cf_matrix = confusion_matrix(test_Y, final_preds)

axis4 = fig.add_subplot(5, 1, 5)
sns.heatmap(cf_matrix, annot = True,ax=axis4)
axis4.set_title("Confusion Matrix for Combined Model on Test Dataset")

symptoms = X.columns.values

#Creating a symptom index dictionary to encode the
#input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = value
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}

#Input: string containing symptoms separated by commas
#Output: Generated predictions by models
def predictDisease(symptoms):
	symptoms = symptoms
	#creating input data for the models
	input_data = [0] * len(data_dict["symptom_index"])
	for symptom in symptoms:
		index = data_dict["symptom_index"][symptom]
		input_data[index] = 1
		
	#reshaping the input data and converting it
	#into suitable format for model predictions
	input_data = np.array(input_data).reshape(1,-1)
	
	#generating individual outputs
	rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
	dt_prediction = data_dict["predictions_classes"][final_dt.predict(input_data)[0]]
	knn_prediction = data_dict["predictions_classes"][final_knn.predict(input_data)[0]]
	
	#making final prediction by taking mode of all predictions
	final_prediction = mode([rf_prediction, dt_prediction, knn_prediction])[0][0]
	predictions = {
		"rf_model_prediction": rf_prediction,
		"decision_tree_prediction": dt_prediction,
		"knn_prediction": knn_prediction,
		"final_prediction":final_prediction
	}
	return predictions
def detail():
 return arr
def plot_png():
 return fig
