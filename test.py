from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
import numpy as np

num_pipeline = make_pipeline([
    SimpleImputer(strategy="median"),
    StandardScaler()
])

cat_pipeline= make_pipeline([
    SimpleImputer(strategy="most_frquent"),
    OneHotEncoder(handle_unknown="ignore")
])

num_attribs = ["longitude","latitude","housing_num_age","total_rooms","total_bedrooms","population","households","median_income"]
cat_attribs = ["ocean_proximity"]

preprocessing = ColumnTransformer([
   ("num", num_pipeline,num_attribs ),
    ("cat",cat_pipeline,cat_attribs)
    
])

num = Pipeline([
    "impute", SimpleImputer(strategy="median"),
    "standardize", StandardScaler()
])


def column_ratio(X):
    return X[:,[0]] / X[:,[1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(column_ratio, feature_names_out=ratio_name ),
    StandardScaler()
)

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log , feature_names_out="one-to-one"),
    StandardScaler()
)

default_num_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler()
)

preprocessing = ColumnTransformer([
    ("", column_ratio, [])
])