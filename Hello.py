"""
Description

The dataset contains information on 13,932 single-family homes sold in Miami in 2016. Besides 
publicly available information, the dataset creator Steven C. Bourassa has added distance 
variables, aviation noise as well as latitude and longitude.

Original dataset source: https://www.openml.org/search?type=data&sort=runs&id=43093&status=active
Dataset downloaded from: https://www.kaggle.com/datasets/deepcontractor/miami-housing-dataset

This app predicts the sale price ("SALE_PRC") of the houses in Miami in 2016 and the feature
importance to determine the price according to the following miami-housing.csv file columns: 

LND_SQFOOT: land area (square feet)
TOT_LVG_AREA: floor area (square feet)
SPEC_FEAT_VAL: value of special features (e.g., swimming pools) ($)
RAIL_DIST: distance to the nearest rail line (an indicator of noise) (feet)
OCEAN_DIST: distance to the ocean (feet)
WATER_DIST: distance to the nearest body of water (feet)
CNTR_DIST: distance to the Miami central business district (feet)
SUBCNTR_DI: distance to the nearest subcenter (feet)
HWY_DIST: distance to the nearest highway (an indicator of noise) (feet)
age: age of the structure
avno60plus: dummy variable for airplane noise exceeding an acceptable level
structure_quality: quality of the structure
month_sold: sale month in 2016 (1 = jan).

It does not take into account for the prediction the following columns of the mimai-housing.csv file:

PARCELNO: unique identifier for each property. About 1% appear multiple times.
LATITUDE
LONGITUDE
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap # Provides the understanding behind the predictio
import matplotlib.pyplot as plt
import pickle
from PIL import Image
 # Import the required variables from model_builder.py
from model_builder import model, X

def main():

    image = Image.open('miami-pic.jpg')

    st.image(image, use_column_width=True)

    # H1 title
    st.write("""
    # Miami Housing Price Prediction App
    This app predicts the **Miami House Price** for properites in *2016*!
    """)

    # Expander bar with the "About" information
    expander_bar = st.expander("Description")
    expander_bar.markdown("""
    This app predicts the sale price ("SALE_PRC") of the houses in Miami in 2016 and the feature
    importance to determine the price according attributes the user can select from the side bar.
    """)
    st.write("---")

    # Sidebar title
    st.sidebar.header("Please Select Your Desired Parameters")
    st.sidebar.write("---")

    df_full = user_input_features()

    # Display the input parameters
    st.header("Selected Input Parameters")
    st.write(df_full)
    st.write("---")

    # Sale Price Prediction by reading the pkl file created in the model_builder.py file
    classifier = pickle.load(open("short_cleaned_miami_housing.pkl", "rb"))
    prediction = classifier.predict(df_full)
    st.header("Sale Price Prediction")
    st.write(prediction)
    st.write("---")

    # Explaining the model's predictiopn using SHAP values
    # https://github.com/slundberg/shap
    importance = shap.TreeExplainer(model)
    shap_vals = importance.shap_values(X)

    st.header("Item Importance")
    # Sumary Plot
    st.set_option('deprecation.showPyplotGlobalUse', False) # Disable deprecation warning
    plt.title("Item Importance Based on SHAP Values")
    shap.summary_plot(shap_vals, X)
    st.pyplot(bbox_inches = "tight")
    st.write("---")

    plt.title("Item Importance Based on SHAP values (Bar)")
    shap.summary_plot(shap_vals, X, plot_type = "bar")
    st.pyplot(bbox_inches = "tight")
    

# Custom function to handle the user input features
def user_input_features():
    # Variables with the inputs
    LND_SQFOOT  = st.sidebar .slider("Land area (square feet)", X.LND_SQFOOT.min(), X.LND_SQFOOT.max(), int(X.LND_SQFOOT.mean()))
    TOT_LVG_AREA = st.sidebar.slider("Floor area (square feet)", X.TOT_LVG_AREA.min(), X.TOT_LVG_AREA.max(), int(X.TOT_LVG_AREA.mean()))
    RAIL_DIST = st.sidebar.slider("Distance to the nearest rail line (an indicator of noise) (feet)", X.RAIL_DIST.min(), X.RAIL_DIST.max(), X.RAIL_DIST.mean())
    OCEAN_DIST = st.sidebar.slider("Distance to the ocean (feet)", X.OCEAN_DIST.min(), X.OCEAN_DIST.max(), X.OCEAN_DIST.mean())
    WATER_DIST = st.sidebar.slider("Distance to the nearest body of water (feet)", X.WATER_DIST.min(), X.WATER_DIST.max(), X.WATER_DIST.mean())
    CNTR_DIST = st.sidebar.slider("Distance to the Miami central business district (feet)", X.CNTR_DIST.min(), X.CNTR_DIST.max(), X.CNTR_DIST.mean())
    SUBCNTR_DI = st.sidebar.slider("Distance to the nearest subcenter (feet)", X.SUBCNTR_DI.min(), X.SUBCNTR_DI.max(), X.SUBCNTR_DI.mean())
    HWY_DIST = st.sidebar.slider("Distance to the nearest highway (an indicator of noise) (feet)", X.HWY_DIST.min(), X.HWY_DIST.max(), X.HWY_DIST.mean())
    age = st.sidebar.slider("Age of the structure", X.age.min(), X.age.max(), round(X.age.mean()))
    structure_quality = st.sidebar.slider("structure quality", X.structure_quality.min(), X.structure_quality.max(), round(X.structure_quality.mean()))
    avno60plus = st.sidebar.selectbox("Airpor sound over 60db / Yes = 1, No = 0", (1, 0))
    month_sold = st.sidebar.selectbox("Month Sold", (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    SPEC_FEAT_VAL = st.sidebar.selectbox("Special Features Value / Yes = 1, No = 0", (1, 0))
    # Dictionary with the input data
    data = {"LND_SQFOOT": LND_SQFOOT,
            "TOT_LVG_AREA": TOT_LVG_AREA,
            "RAIL_DIST": RAIL_DIST ,
            "OCEAN_DIST": OCEAN_DIST,
            "WATER_DIST": WATER_DIST,
            "CNTR_DIST": CNTR_DIST ,
            "SUBCNTR_DI": SUBCNTR_DI,
            "HWY_DIST": HWY_DIST,
            "age": age,
            "structure_quality": structure_quality,
            "avno60plus": avno60plus ,
            "month_sold": month_sold ,
            "SPEC_FEAT_VAL": SPEC_FEAT_VAL
    },
    
    # Creates and return the data frame
    features = pd.DataFrame(data, index = ["Value"])
    return features


if __name__ == '__main__':
	main()