import pandas as pd
import numpy as np
import joblib
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from PIL import Image


st.title("Disease Predictions Using ML Model")
st.subheader("Overview of DataSet we are working On")


data=pd.read_csv("Training.csv").dropna(axis=1)
encoder=LabelEncoder()

st.dataframe(data)

data['prognosis']=encoder.fit_transform(data["prognosis"])
x=data.iloc[:,:-1]

symptoms=x.columns.values

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
     
    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
         
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
     
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][final_nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][final_svm_model.predict(input_data)[0]]
     
    # making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, nb_prediction, svm_prediction],keepdims=True)[0][0]
    return final_prediction


final_svm_model=joblib.load('final_svm_model')
final_rf_model=joblib.load('final_rf_model')
final_nb_model=joblib.load('final_nb_model')

st.header("Confusion Matrix Of Our Model")
image = Image.open('Confusion.png')

st.image(image, caption='Confusion Matrix')


if st.sidebar.checkbox('Predict Disease Based on the Input'):
    options = st.sidebar.multiselect(
    'What are your favorite colors',
    ['Itching','Skin Rash','Nodal Skin Eruptions','Continuous Sneezing','Shivering','Chills'
,'Joint Pain','Stomach Pain','Acidity','Ulcers On Tongue'
,'Muscle Wasting'
,'Vomiting'
,'Burning Micturition'
,'Spotting  urination'
,'Fatigue'
,'Weight Gain'
,'Anxiety'
,'Cold Hands And Feets'
,'Mood Swings'
,'Weight Loss'
,'Restlessness'
,'Lethargy'
,'Patches In Throat'
,'Irregular Sugar Level'
,'Cough'
,'High Fever'
,'Sunken Eyes'
,'Breathlessness'
,'Sweating'
,'Dehydration'
,'Indigestion'
,'Headache'
,'Yellowish Skin'
,'Dark Urine'
,'Nausea'
,'Loss Of Appetite'
,'Pain Behind The Eyes'
,'Back Pain'
,'Constipation'
,'Abdominal Pain'
,'Diarrhoea'
,'Mild Fever'
,'Yellow Urine'
,'Yellowing Of Eyes'
,'Acute Liver Failure'
,'Fluid Overload'
,'Swelling Of Stomach'
,'Swelled Lymph Nodes'
,'Malaise'
,'Blurred And Distorted Vision'
,'Phlegm'
,'Throat Irritation'
,'Redness Of Eyes'
,'Sinus Pressure'
,'Runny Nose'
,'Congestion'
,'Chest Pain'
,'Weakness In Limbs'
,'Fast Heart Rate'
,'Pain During Bowel Movements'
,'Pain In Anal Region'
,'Bloody Stool'
,'Irritation In Anus'
,'Neck Pain'
,'Dizziness'
,'Cramps'
,'Bruising'
,'Obesity'
,'Swollen Legs'
,'Swollen Blood Vessels'
,'Puffy Face And Eyes'
,'Enlarged Thyroid'
,'Brittle Nails'
,'Swollen Extremeties'
,'Excessive Hunger'
,'Extra Marital Contacts'
,'Drying And Tingling Lips'
,'Slurred Speech'
,'Knee Pain'
,'Hip Joint Pain'
,'Muscle Weakness'
,'Stiff Neck'
,'Swelling Joints'
,'Movement Stiffness'
,'Spinning Movements'
,'Loss Of Balance'
,'Unsteadiness'
,'Weakness Of One Body Side'
,'Loss Of Smell'
,'Bladder Discomfort'
,'Foul Smell Of urine'
,'Continuous Feel Of Urine'
,'Passage Of Gases'
,'Internal Itching'
,'Toxic Look (typhos)'
,'Depression'
,'Irritability'
,'Muscle Pain'
,'Altered Sensorium'
,'Red Spots Over Body'
,'Belly Pain'
,'Abnormal Menstruation'
,'Dischromic  Patches'
,'Watering From Eyes'
,'Increased Appetite'
,'Polyuria'
,'Family History'
,'Mucoid Sputum'
,'Rusty Sputum'
,'Lack Of Concentration'
,'Visual Disturbances'
,'Receiving Blood Transfusion'
,'Receiving Unsterile Injections'
,'Coma'
,'Stomach Bleeding','Distention Of Abdomen','History Of Alcohol Consumption','Fluid Overload.1','Blood In Sputum','Prominent Veins On Calf','Palpitations','Painful Walking','Pus Filled Pimples','Blackheads','Scurring','Skin Peeling','Silver Like Dusting','Small Dents In Nails','Inflammatory Nails','Blister','Red Sore Around Nose','Yellow Crust Ooze','Fungal infection'])


    if (len(options)!=0):
        finalstr=','.join(options)
        result=predictDisease(finalstr)
        st.header("Disease Predicted")
        st.write("I'm Sorry But,You are Diagonised with ",result)


    