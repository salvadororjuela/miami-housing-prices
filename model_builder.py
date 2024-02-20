"""
This python file creates the .pkl file to be used by miami_housing_prices.py as model
and avoid charging it for every prediction.
"""
import pandas as pd
import pickle # Reads the pickle file produced in model_builder.py
from sklearn.ensemble import RandomForestRegressor

miami_df = pd.read_csv("cleaned_miami_housing.csv")

# Create a copy of the dataframe into the df variable
df = miami_df.copy()
# target will be the variable/parameter to predict the price
target = "SALE_PRC"

# Loads the Miami data and determine independent and target variables
miami_csv = pd.read_csv("cleaned_miami_housing.csv")
X = pd.DataFrame(miami_csv[["LND_SQFOOT", "TOT_LVG_AREA", "RAIL_DIST", "OCEAN_DIST", 
                            "WATER_DIST", "CNTR_DIST", "SUBCNTR_DI", "HWY_DIST", "age", 
                            "structure_quality", "avno60plus", "month_sold", "SPEC_FEAT_VAL"]])
Y = pd.DataFrame(miami_csv["SALE_PRC"])

# Build and train the Regression Model
model = RandomForestRegressor()
# Y.values.ravel() corrects the following error:
# A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
model.fit(X, Y.values.ravel())

# Saving the model
pickle.dump(model, open("short_cleaned_miami_housing.pkl", "wb"))
