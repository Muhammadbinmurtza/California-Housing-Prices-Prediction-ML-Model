from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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


