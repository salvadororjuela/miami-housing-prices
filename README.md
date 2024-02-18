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

For the purpose of being able to see the application result and functionality, the model_builder.py
file uses the first 1500 rows, instead of using the whole dataframe, which would result in a long 
time awaiting period.
