import pandas as pd
import numpy as np
import copy
import math
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

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

