import numpy as np
import pickle
import streamlit as st
import sklearn

loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def concrete_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    return(prediction)

def main():
    
    st.title("Concrete Strength Prediction App")
    
    Cement = st.text_input('Cement')
    BlastFurnaceSlag = st.text_input('BlastFurnaceSlag')
    FlyAsh = st.text_input('FlyAsh')
    Water = st.text_input('Water')
    Superplasticizerent = st.text_input('Superplasticizer')
    CoarseAggregate = st.text_input('CoarseAggregate')
    FineAggregate = st.text_input('FineAggregate')
    Age = st.text_input('Age')
    
    strength = ''
    
    if st.button("Find Strength"):
        strength = concrete_prediction([Cement, BlastFurnaceSlag, FlyAsh, Water, Superplasticizerent, CoarseAggregate, FineAggregate, Age])
    
    st.success(strength)
    
if __name__ == '__main__':
    main()