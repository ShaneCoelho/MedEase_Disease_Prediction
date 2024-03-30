import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import warnings
from statistics import mode
import joblib

train_data=pd.read_csv("notebook/Training.csv").dropna(axis=1)
test_data=pd.read_csv("notebook/Testing.csv").dropna(axis=1)

encoder=LabelEncoder()
train_data['prognosis']=encoder.fit_transform(train_data['prognosis'])
test_data['prognosis']=encoder.fit_transform(test_data['prognosis'])

x=train_data.iloc[:,:-1]
y=train_data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=24)

# svmModelFit = SVC()
# nbModelFit = GaussianNB()
# rfModelFit = RandomForestClassifier(random_state=18)

# svmModelFit.fit(x, y)
# nbModelFit.fit(x, y)
# rfModelFit.fit(x, y)

test_x = test_data.iloc[:, :-1]
test_y = test_data.iloc[:, -1]

svmModelFit=joblib.load('notebook/svmModel.joblib')
nbModelFit=joblib.load('notebook/nbModel.joblib')
rfModelFit=joblib.load('notebook/rfModel.joblib')

svmPredicts = svmModelFit.predict(test_x)
nbModelPredicts = nbModelFit.predict(test_x)
rfModelPredicts = rfModelFit.predict(test_x)

symptoms = x.columns.values

# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
	symptom = " ".join([i.capitalize() for i in value.split("_")])
	symptom_index[symptom] = index

data_dict = {
	"symptom_index":symptom_index,
	"predictions_classes":encoder.classes_
}


def predictDisease(symptoms):
	symptoms = symptoms.split(",")
	
	input_data = [0] * len(data_dict["symptom_index"])
	for symptom in symptoms:
		index = data_dict["symptom_index"][symptom]
		input_data[index] = 1
		

	input_data = np.array(input_data).reshape(1,-1)
	
	# generating individual outputs
	rf_prediction = data_dict["predictions_classes"][rfModelFit.predict(input_data)[0]]
	nb_prediction = data_dict["predictions_classes"][nbModelFit.predict(input_data)[0]]
	svm_prediction = data_dict["predictions_classes"][svmModelFit.predict(input_data)[0]]
	
	# making final prediction by taking mode of all predictions
	final_prediction = mode([rf_prediction, nb_prediction, svm_prediction])
	predictions = {
		"rf_model_prediction": rf_prediction,
		"naive_bayes_prediction": nb_prediction,
		"svm_model_prediction": svm_prediction,
		"final_prediction":final_prediction
	}
	return predictions

# Testing the function
# warnings.filterwarnings("ignore", category=UserWarning)
# list_of_symptoms = ["Polyuria", "Increased Appetite", "Excessive Hunger", "Obesity", "Skin Rash", "Blurred And Distorted Vision", "Fatigue"]
# test_symptoms = ",".join(list_of_symptoms)
# test_predictions = predictDisease(test_symptoms)
# print(test_predictions)
