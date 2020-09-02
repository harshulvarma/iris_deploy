import streamlit as st
import pandas as pd
import pickle

st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!

--using randomm forest classifier--

""")

st.header('Input your parameters')

def user_input_features():
    sepal_length = st.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

clf = pickle.load(open('iris_model_weights.pkl','rb'))

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

target_names = ['setosa','versicolor','virginca']
st.subheader('Prediction')
st.write(target_names[prediction[0]])
# st.write(prediction)

st.subheader('Prediction Probabilities')

st.text('For setosa:')
st.write(prediction_proba[0][0]*100)

st.text('For versicolor:')
st.write(prediction_proba[0][1]*100)

st.text('For virginca:')
st.write(prediction_proba[0][2]*100)