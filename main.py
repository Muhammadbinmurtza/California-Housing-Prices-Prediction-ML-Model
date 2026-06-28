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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

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
encoder = OrdinalEncoder()
housing_cat_encoded = encoder.fit_transform(housing_cat)

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

#Cluster similarity 

class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None): 
        self.kmeans_ = KMeans(
            self.n_clusters,
            random_state=self.random_state
        )
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self

    def transform(self, X):
        return rbf_kernel(
            X,
            self.kmeans_.cluster_centers_,
            gamma=self.gamma
        )

    def get_feature_names_out(self, names=None):
        return [
            f"Cluster {i} similarity"
            for i in range(self.n_clusters)
        ]
    
# pipeline for transformation Page #83
num_pipeline = Pipeline([
    ("impute" , SimpleImputer(strategy="median")),
    ("standardize", StandardScaler()),
])

housing_num_prepared = num_pipeline.fit_transform(housing_num)
housing_num_prepared[:2].round(2)

df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, columns=num_pipeline.get_feature_names_out(), index=housing_num.index
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
    (cat_pipeline , make_column_selector(dtype_include=object))
)

housing_prepared = preprocessing.fit_transform(housing)

housing_prepared_df1 = pd.DataFrame(
    housing_prepared, columns=preprocessing.get_feature_names_out(),index=housing.index
)

print(housing_prepared_df1)

def column_ratio(X):
    return X[:,[0]] / X[:,[1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler()
    )

log_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log, feature_names_out="one-to-one"),
    StandardScaler()
)

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1, random_state=42)

default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())

preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)

housing_prepared = preprocessing.fit_transform(housing)
print(housing_prepared.shape)

print(preprocessing.get_feature_names_out())

lin_reg = make_pipeline(
    preprocessing, 
    LinearRegression()
)

lin_reg.fit(housing, housing_labels)

housing_prediction = lin_reg.predict(housing)
print(housing_prediction[:5].round(-2))

print(housing_labels.iloc[:5].values)

lin_rmse = root_mean_squared_error(housing_labels, housing_prediction)

print(lin_rmse)

tree_reg = make_pipeline(preprocessing, 
                        DecisionTreeRegressor(random_state=42))

tree_reg.fit(housing, housing_labels)

housing_prediction = tree_reg.predict(housing)

tree_rmse = root_mean_squared_error(housing_labels, housing_prediction)
print(tree_rmse)  # provided zero error

#checking the prediction on 10 cv sets

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)

print(pd.Series(tree_rmses).describe())
'''
prediction is not good a little better than linear regression but still bad
count       10.000000
mean     67331.259647
std       2411.905355
min      62697.335860
25%      66006.108594
50%      67430.321931
75%      68628.596244
max      71055.427006
'''

# let's try random forest 

forest_reg = make_pipeline(preprocessing,
                           RandomForestRegressor(random_state=42))

forest_rmses = -cross_val_score(forest_reg,
                               housing, 
                               housing_labels,
                               scoring="neg_root_mean_squared_error",
                               cv=10)

print(pd.Series(forest_rmses).describe())


## Fine tune the model , random forest algorithm

full_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("random_forest", RandomForestRegressor(random_state=42))
])

param_grid = [
    {
        'preprocessing__geo__n_clusters' : [5,8,10],
        'random_forest__max_features' : [4,6,8,]
    },
    {
        'preprocessing__geo__n_clusters' : [10,15],
        'random_forest__max_features' : [6,8,10]
    }
]

grid_search = GridSearchCV(full_pipeline, param_grid, cv= 3, scoring= "neg_root_mean_squared_error")

grid_search.fit(housing, housing_labels)

print(grid_search.best_params_)

cv_res = pd.DataFrame(grid_search.cv_results_)
cv_res.sort_values(by="mean_test_score", ascending=False, inplace=True)
cv_res = cv_res[["param_preprocessing__geo__n_clusters",
                 "param_random_forest__max_features", "split0_test_score",
                 "split1_test_score", "split2_test_score", "mean_test_score"]]
score_cols = ["split0", "split1", "split2", "mean_test_rmse"]
cv_res.columns = ["n_clusters", "max_features"] + score_cols
cv_res[score_cols] = -cv_res[score_cols].round().astype(np.int64)

print(cv_res.head())
