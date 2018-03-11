import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt
import seaborn as sns

## read in all 5 csv's in file
boros = ['manhattan', 'bronx', 'brooklyn', 'queens', 'statenisland']

dat_frame = []
for boro in boros:
    df = pd.read_csv('Data/rollingsales_' + boro +'.csv',skiprows=4)
    dat_frame.append(df)
dat = pd.concat(dat_frame)


## check data
dat.dtypes
feature_list = list(dat);
feature_list_new = ['BOROUGH', 'NEIGHBORHOOD', 'BUILDING_CLASS_CATEGORY',
       'TAX_CLASS_AT_PRESENT', 'BLOCK', 'LOT', 'EASE_MENT',
       'BUILDING_CLASS_AT_PRESENT', 'ADDRESS', 'APARTMENT_NUMBER',
       'ZIP_CODE', 'RESIDENTIAL_UNITS','COMMERCIAL UNITS', 'TOTAL_UNITS',
       'LAND_SQUARE_FEET', 'GROSS_SQUARE_FEET', 'YEAR_BUILT',
       'TAX_CLASS_AT_TIME_OF_SALE', 'BUILDING_CLASS_AT_TIME_OF_SALE',
       'SALE_PRICE', 'SALE_DATE'];
dat.columns = feature_list_new;


## data preprocessing
dat.drop_duplicates(inplace=True) ##drop duplicates input

## only include residential property
dat = dat[(dat.TAX_CLASS_AT_TIME_OF_SALE == 1) | (dat.TAX_CLASS_AT_TIME_OF_SALE == 2)]
del dat['TAX_CLASS_AT_TIME_OF_SALE']

## keep 'building class at present' == 'building_class at time of sale'
# & 'tax class at present' == 'tax class at time of sale'
dat = dat[(dat.BUILDING_CLASS_AT_PRESENT == dat.BUILDING_CLASS_AT_TIME_OF_SALE)]



## keep only informative features
dat = dat[['BOROUGH','NEIGHBORHOOD','RESIDENTIAL_UNITS', 'TOTAL_UNITS',
       'LAND_SQUARE_FEET', 'GROSS_SQUARE_FEET', 'YEAR_BUILT', 'BUILDING_CLASS_AT_TIME_OF_SALE',
       'SALE_PRICE', 'SALE_DATE']]


## clean response variable: sale price
dat['SALE_PRICE'] = dat.SALE_PRICE.str.strip();
dat = dat[(dat.SALE_PRICE != '-')];  ## remove missing sale price
dat['SALE_PRICE'] = dat.SALE_PRICE.str.replace(',', '');
dat['SALE_PRICE'] = pd.to_numeric(dat.SALE_PRICE, errors='force')
dat = dat[(dat.SALE_PRICE >= np.percentile(dat.SALE_PRICE,2)) &
          (dat.SALE_PRICE <= np.percentile(dat.SALE_PRICE,98))] ## remove response outliers
dat['LOG_SALE_PRICE'] = np.log(dat['SALE_PRICE'])

## clean features
boro_map =  {1:'Manhattan',  2: 'Bronx',3: 'Brooklyn',4: 'Queens', 5:'StatenIsland'}
dat['BOROUGH'] = dat['BOROUGH'].map(boro_map)

dat = dat[dat.YEAR_BUILT > 1700]
#plt.hist(dat.YEAR_BUILT) #have 3 peaks
dat['YEAR_BUILT_Q'] = pd.cut(dat['YEAR_BUILT'], [1700,1920,1960,2020], labels=['1700-1920','1920-1960','1960-2018'])

dat['LAND_SQUARE_FEET'] = dat.LAND_SQUARE_FEET.str.strip();
dat['GROSS_SQUARE_FEET'] = dat.GROSS_SQUARE_FEET.str.strip();
dat = dat[dat.LAND_SQUARE_FEET != '-'];  ## replace '-' with NA
#dat = dat[dat.GROSS_SQUARE_FEET != '-'];  ## replace '-' with NA
dat['LAND_SQUARE_FEET'] = dat.LAND_SQUARE_FEET.str.replace(',', '');
#dat['GROSS_SQUARE_FEET'] = dat.LAND_SQUARE_FEET.str.replace(',', '');
dat['LAND_SQUARE_FEET'] = pd.to_numeric(dat.LAND_SQUARE_FEET, errors='force');
#dat['GROSS_SQUARE_FEET'] = pd.to_numeric(dat.GROSS_SQUARE_FEET, errors='force')
dat = dat[(dat.LAND_SQUARE_FEET >= np.percentile(dat.LAND_SQUARE_FEET,2)) &
          (dat.LAND_SQUARE_FEET <= np.percentile(dat.LAND_SQUARE_FEET,98))] ## remove response outliers
del dat['GROSS_SQUARE_FEET']


dat['TOTAL_UNITS'] = dat.TOTAL_UNITS.str.strip();
dat = dat[dat.TOTAL_UNITS != '-'];  ## replace '-' with NA
dat['TOTAL_UNITS'] = pd.to_numeric(dat.TOTAL_UNITS, errors='raise');
dat = dat[(dat.TOTAL_UNITS >= np.percentile(dat.TOTAL_UNITS,2)) &
          (dat.TOTAL_UNITS <= np.percentile(dat.TOTAL_UNITS,98))]



dat['BUILDING_CLASS_AT_TIME_OF_SALE'] = dat.BUILDING_CLASS_AT_TIME_OF_SALE.str.strip();
dat['BUILDING_CLASS_AT_TIME_OF_SALE'] = dat.BUILDING_CLASS_AT_TIME_OF_SALE.apply(lambda x: x[0])
#dat.BUILDING_CLASS_AT_TIME_OF_SALE.value_counts()


dat['SALE_DATE'] = pd.to_datetime(dat.SALE_DATE,errors='raise')
dat['MONTH'] = dat['SALE_DATE'].dt.month
season_map =  {'Winter':[12,1,2], 'Spring': [3,4,5],'Summer':[6,7,8],'Fall': [9,10,11]}
reversed_season_map = {val: key for key in season_map for val in season_map[key]}
dat['SEASON'] = dat['MONTH'].map(reversed_season_map)

## delete features would not be used
del dat['SALE_PRICE']
del dat['YEAR_BUILT']
del dat['RESIDENTIAL_UNITS']
del dat['SALE_DATE']
del dat['MONTH']

## descriptive

## log(sale price)
plt.hist(dat.LOG_SALE_PRICE,bins=20)
plt.xlabel('Log(Sale Price)')
plt.ylabel('Frequencies')
plt.savefig('Report/Fig/log_sale.pdf',dpi=900)

## log sale price by year
sns.boxplot(x='YEAR_BUILT_Q',y='LOG_SALE_PRICE',data=dat,showfliers=False)
plt.xlabel('Building Year')
plt.ylabel('Log(Sale Price)')
plt.savefig('Report/Fig/year_sale_price.pdf',dpi=900)

## log sale price by borough
sns.boxplot(x='BOROUGH',y='LOG_SALE_PRICE',data=dat,showfliers=False)
plt.xlabel('Boroughs')
plt.ylabel('Log(Sale Price)')
plt.savefig('Report/Fig/borough_sale_price.pdf',dpi=900)

## log sale price by building class
sns.boxplot(x='BUILDING_CLASS_AT_TIME_OF_SALE',y='LOG_SALE_PRICE',data=dat,showfliers=False)
plt.xlabel('Building Class')
plt.ylabel('Log(Sale Price)')
plt.savefig('Report/Fig/building_class_sale_price.pdf',dpi=900)

## log sale price by total units
sns.boxplot(x='TOTAL_UNITS',y='LOG_SALE_PRICE',data=dat,showfliers=False)
plt.xlabel('Total Units')
plt.ylabel('Log(Sale Price)')
plt.savefig('Report/Fig/units_class_sale_price.pdf',dpi=900)

## log sale price by season
sns.boxplot(x="SEASON", y="LOG_SALE_PRICE", data=dat,showfliers=False)
plt.xlabel('Season')
plt.ylabel('Log(Sale Price)')
plt.savefig('Report/Fig/season_sale_price.pdf',dpi=900)

## log sale price by neighborhood
sns.boxplot(x="NEIGHBORHOOD", y="LOG_SALE_PRICE", data=dat,showfliers=False)
plt.xlabel('Neighborhood')
plt.ylabel('Log(Sale Price)')
plt.savefig('Report/Fig/neighbor_sale_price.pdf',dpi=900)



## log sale price vs land square feet
sns.regplot(x="LAND_SQUARE_FEET", y="LOG_SALE_PRICE", data=dat, lowess=True)
plt.xlabel('Land Square Feet')
plt.ylabel('Log(Sale Price)')
plt.savefig('Report/Fig/sqfeet_price.pdf',dpi=900)




# Encoding categorical data

categorical_features_list = ['BOROUGH', 'NEIGHBORHOOD','BUILDING_CLASS_AT_TIME_OF_SALE',
                             'YEAR_BUILT_Q','SEASON']
dat_new = pd.get_dummies(dat)
dat_new.drop(labels=['BOROUGH_Bronx','NEIGHBORHOOD_AIRPORT LA GUARDIA','BUILDING_CLASS_AT_TIME_OF_SALE_V',
                     'YEAR_BUILT_Q_1700-1920', 'SEASON_Fall'], axis=1, inplace=True)


y = dat_new['LOG_SALE_PRICE'].values
del dat_new['LOG_SALE_PRICE']
X = dat_new.values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


reg = RandomForestRegressor(criterion='mse',random_state=0)

#report best scores
def report(results, num = 5):
    for i in range(1, num + 1):
        cv_res = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in cv_res:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("**************************")

param_grid = {
              "max_features": ['auto','log2'],
              "min_samples_leaf": [10,20,50],
              "n_estimators": [50,100]
            }

# run grid search
start = time.time()
grid_search = GridSearchCV(reg, cv=5,param_grid=param_grid)
grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)

#print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time.time() - start, len(grid_search.cv_results_['params'])))
#report(grid_search.cv_results_)


print(grid_search.score(X_test,y_test))
y_test_pred = grid_search.predict(X_test)


reg_RF = RandomForestRegressor(criterion='mse',random_state=0,max_features='auto',
                               min_samples_leaf=10,n_estimators=100)
reg_RF.fit(X_train,y_train)
reg_RF.score(X_test,y_test)
print ("Features sorted by their score:")
imp_array = np.array(sorted(zip(map(lambda x: round(x, 2),reg_RF.feature_importances_),list(dat_new)),
             reverse=True))
print (imp_array)

size = 10

plt.title('Important Features')
plt.barh(range(imp_array[0:size,0].shape[0] ), imp_array[0:size,0], color='green', align='center')
plt.yticks(range(imp_array[0:size,1].shape[0] - 1), imp_array[0:size,1], fontsize=4,rotation=45)
plt.xlabel('Importance Score')
plt.savefig('Report/Fig/imp_features.pdf',dpi=900)




##### ANN Exporation
# Tuning the ANN
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler

# Feature Scaling
sc = StandardScaler()
X_train_ann = sc.fit_transform(X_train)
X_test_ann = sc.transform(X_test)

def build_regressor(optimizer):
    regressor = Sequential()
    regressor.add(Dense(units = 100, kernel_initializer = 'normal', activation = 'relu', input_dim = 244))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 50, kernel_initializer = 'normal', activation = 'relu'))
    regressor.add(Dropout(0.2))
    regressor.add(Dense(units = 1, kernel_initializer = 'normal'))
    regressor.compile(optimizer = optimizer, loss = 'mean_squared_error')
    return regressor


regressor = KerasRegressor(build_fn = build_regressor, epochs=100)
parameters = {'batch_size': [25, 32],
              'optimizer': ['adam', 'rmsprop']}
grid_search_ann = GridSearchCV(estimator = regressor,
                           param_grid = parameters,
                           scoring = 'mean_squared_error',
                           cv = 5)
grid_search_ann = grid_search_ann.fit(X_train_ann, y_train)
best_parameters = grid_search_ann.best_params_

y_test_pred_ann = grid_search_ann.predict(X_test_ann)
from sklearn.metrics import r2_score
r2_score(y_test,y_test_pred_ann)