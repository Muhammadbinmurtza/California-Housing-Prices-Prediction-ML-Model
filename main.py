import pandas as pd
import numpy as np
import copy
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder,FunctionTransformer,OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor, make_column_selector,make_column_transformer,ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline,make_pipeline


data = pd.read_csv("documents/housing.csv")

df = data.head()
print(df)

print(data.info())
print(data.describe())

print(data["ocean_proximity"].value_counts())

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

data.hist(bins=50, figsize=(12,8))
# plt.show()

#creating a test set

def shuffle_and_split_data(data , test_ratio):
    shuffle_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = shuffle_and_split_data(data, 0.2)
print(len(train_set))
print(len(test_set))
 

housing = copy.deepcopy(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# print(test_set["total_bedrooms"].isnull().sum())

housing['total_bedrooms']=housing["total_bedrooms"].fillna(housing['total_bedrooms'].mean())
print(housing['total_bedrooms'].isnull().sum())

housing['income_cat'] = pd.cut(housing['median_income'], bins=[0.,1.5,3.0,4.5,6.0, np.inf],
                               labels=[1,2,3,4,5])

housing['income_cat'].value_counts().sort_index().plot.bar(rot=0,grid=True)
plt.xlabel("income category")
plt.ylabel("number of districts")
plt.savefig("bar chart of income category")
plt.show()

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing , housing['income_cat']):
    strat_train_n = housing.iloc[train_index]
    strat_test_n = housing.iloc[test_index]
    strat_splits.append([strat_train_n, strat_test_n])
    strat_train_set,strat_test_set = strat_splits[0]

strat_train_n , strat_test_n = train_test_split(housing , test_size=0.2, stratify=housing['income_cat'],random_state=42)

print(strat_test_set['income_cat'].value_counts()/len(strat_test_set))

housing.drop(columns=['income_cat'],inplace=True)

# print(housing.head())

# print(strat_train_set)

housing = strat_train_set.copy()

housing.plot(kind='scatter',x= 'longitude',y='latitude',grid=True)
plt.savefig("longitude and latitude plot")
plt.show()  # bad visualization plot

housing.plot(kind='scatter', x='longitude',y="latitude", grid=True, alpha=0.2)
plt.savefig("update plot using alpha")
plt.show()  #better visualization

housing.plot(kind='scatter', x='longitude', y='latitude',s=housing['population']/100, c='median_house_value',
             cmap='jet',colorbar=True, legend=True, sharex=False,figsize=(10,7))
plt.savefig("best plot using cmap jet and c and s")
plt.show()


# the correlation cofficient (pearson's r)
corr_matrix = housing.corr(numeric_only=True)

print(corr_matrix["median_house_value"].sort_values(ascending=False))

attributes = ['median_house_value','median_income','total_rooms','housing_median_age']
scatter_matrix(housing[attributes],figsize=(12,8))
plt.savefig("standard correlation cofficient figure")
plt.show()


housing['rooms_per_house'] = housing['total_rooms']/housing['households']
housing['bedrooms_ratio'] = housing['total_bedrooms']/housing['total_rooms']
housing['people_per_house'] = housing['population'] / housing['households']

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix['median_house_value'].sort_values(ascending=False))

housing = strat_train_set.drop("median_house_value", axis=1)  # deleting the y column
housing_labels = strat_train_set['median_house_value'].copy() # that is what model is gonna predict

median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median, inplace=True)


imputer = SimpleImputer(strategy="median")

housing_num = housing.select_dtypes(include=[np.number])

imputer.fit(housing_num)

X= imputer.transform(housing_num)

housing_tr = pd.DataFrame(X
                          , columns=housing_num.columns, index=housing_num.index)

isolation_forest = IsolationForest(random_state=42)  # to remove the outliers from the dataset
outliers_pred = isolation_forest.fit_predict(X)

housing_cat = housing[["ocean_proximity"]]

housing_cat_encoded = OrdinalEncoder.fit_transform(housing_cat)

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

#converting sparse array to numpy array 
print(housing_cat_1hot.toarray())

#min max scaler for feature scaling

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_max_scaler = min_max_scaler.fit_transform(housing_num)

#standard scaler for feature scaling

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_num)

# showing the similiarity of a housing age with 35 

age_simi_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

#predicting the value with linear regression algorithm
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame()) # converting the dataset into dataframe

model = LinearRegression()
model.fit(housing[["median_income"]], scaled_labels)
some_new_data = housing[["median_income"]].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)

# fitting and scaling in one line

model = TransformedTargetRegressor(LinearRegression(),transformer=StandardScaler())
model.fit(housing[["median_income"]],housing_labels)
predictions = model.predict(some_new_data)

# creating your own transformer named as log transformer

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)

log_pop = log_transformer.transform(housing[["population"]])

# pipeline for transformation Page #83
num_pipeline = Pipeline([
    ("impute" , SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:2].round(2)

df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(), index=housing_num.index()
)

# now using the column transformer to handle both numerical and categorical values in a pipeline

num_attribs = ["longitude","latitude","housing_num_age","total_rooms","total_bedrooms","population","households","median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs)
])

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),
    (cat_pipeline , make_column_selector(object))
)

housing_prepared = preprocessing.fit_transform(housing)

