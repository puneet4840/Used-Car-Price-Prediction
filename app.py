# Creating Web app using Streamlit python library.
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import webbrowser


## Creating Nav Bar
nav=st.sidebar.radio("Navigation",("Home","Prediction","About"))

if nav=="Home":
    st.markdown("""
    <style>
    .big-font {
        font-size:60px !important;
        color: orange;
        text-align: left;
        font-weight:50;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Used Car Price Predictor</p>', unsafe_allow_html=True)

    st.image("Car Price Prediction pic.jpeg",width=600)

    # Welcome

    st.markdown("""
    <style>
    .welcome {
        font-size:35px !important;
        color: black;
        text-align: left;
        font-weight:50;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="welcome">WELCOME</p>', unsafe_allow_html=True)

    # Description

    st.markdown("""
    <style>
    .description {
        font-size:20px !important;
        color: black;
        text-align: left;
        font-weight:50;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="description">This is a Web Application that predicts the price of used cars based on some features of the car.</p>', unsafe_allow_html=True)




elif nav=="Prediction":
    
    st.markdown("""
    <style>
    .big-font {
        font-size:60px !important;
        color: orange;
        text-align: center;
        font-weight:50;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="big-font">Used Car Price Predictor</p>', unsafe_allow_html=True)


    # Getting Data from csv file
    car=pd.read_csv("Cleaned data of Main file 3n.csv")

    # Loading the ML Model
    ml_model=pickle.load(open('Used Car Price Prediction Model.pkl','rb'))


    ## Sidebar
    ## Getting input data from web app for our model.

    ## Car_Company

    company=st.selectbox("Select Company",(sorted(car['company'].unique())))

    ## Car_Model

    model=[]
    model=[c_m for c_m in car['name'].unique() if company in c_m]
    car_model=st.selectbox("Select Car Model",(sorted(model)))

    ## Year

    year=st.selectbox("Select Year",(sorted(car['year'].unique(),reverse=True)))

    ## Km_driven

    km_driven=st.text_input("Enter Kms Driven")

    ## Fuel_Type

    fuel_type=st.selectbox("Select Fuel Type",(car['fuel'].unique()))

    ## Transmission

    transmission=st.selectbox("Select Transmission Type",(car['transmission'].unique()))

    ## Owner

    owner=st.selectbox("Select Owner",(car['owner'].unique()))

    ## Prediction
    if st.button("Predict"):
        if km_driven:
            prediction=ml_model.predict(pd.DataFrame([[car_model,company,year,km_driven,fuel_type,transmission,owner]],columns=['name','company','year','km_driven','fuel','transmission','owner']))
            st.success(f"Price:   {str(np.round(prediction[0],2))}")
        else:
            st.warning("Please Enter the Kilometer Driven")


elif nav=="About":
    st.markdown("""
    <style>
    .about {
        font-size:50px !important;
        color: black;
        text-align: left;
        font-weight:50;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="about">Thank You for visiting.</p>', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .myself {
        font-size:35px !important;
        color: black;
        text-align: left;
        font-weight:50;
    }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="myself">This Project is made my Puneet Kumar.</p>', unsafe_allow_html=True)

    st.write("Please Checkout My Github and Linkedin Profile")

    link = '[GitHub](https://github.com/puneet4840)'
    st.markdown(link, unsafe_allow_html=True)

    link1 = '[Linkedin](https://www.linkedin.com/in/puneet-kumar-0916271a3/)'
    st.markdown(link1,unsafe_allow_html=True)
