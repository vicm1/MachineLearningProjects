import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

csv_file_path = r'MachineLearningProjects\MachineLearningModel\teams.csv' # store the file path of the csv in a variable for easier access
teams = pd.read_csv(csv_file_path)

teams = teams[["team","country","year","athletes","age","prev_medals","medals"]] # make data with new columns that show only data needed, removing unnecessary columns. 


#sns.lmplot(x="athletes", y = "medals", data = teams, fit_reg = True, ci= None) # plot graph that shows linear relationship of number of athletes that a country enters to the olympics more medals earned

#sns.lmplot(x="age", y = "medals", data = teams, fit_reg = True, ci= None) # plot graphs that shows how many medals won by age, seems that age doesn't come into large play when winning medals

#teams.plot.hist(y="medals")
#plt.show() # show the plot

teams[teams.isnull().any(axis=1)] # remove missing values, find any rows that has missing values
teams = teams.dropna()
train = teams[teams["year"]<2012].copy() # train the model on this set
test = teams[teams["year"]>=2012].copy() # test set used to see how well the model is doing, don't want to test it using the train set since it already knows the answer

train.shape
test.shape

print(train)
print(test)

reg = LinearRegression() # train a linear regression model

prediction = ["athletes", "prev_medals"] # columns we're going to use to predict our target
target = "medals" # the medals is our target

reg.fit(train[prediction],train["medals"]) # fit the linear regression model

predictions = reg.predict(test[prediction]) # make predictions


print(predictions)

test["predictions"] = predictions
test.loc[test["predictions"]<0, "predictions"] = 0 # index test data frame, find any rows where predictions column is less than 0 and replace that predictions value in that row with a 0. 
test["predictions"] = test["predictions"].round() # round to the nearest whole number
print(test)

error = mean_absolute_error(test["medals"], test["predictions"]) # on avg within 3.3 medlas on how many medals a team actual won
print(error)

print(teams.describe()["medals"])

print(test[test["team"] == "USA"]) 

errors = (test["medals"] - test["predictions"]).abs()
print(errors)

error_by_team = errors.groupby(test["team"]).mean()
print(error_by_team)

medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio = error_by_team/medals_by_team
print(error_ratio)

error_ratio[~pd.isnull(error_ratio)] # remove missing values

error_ratio = error_ratio[np.isfinite(error_ratio)] # remove infinite values

print(error_ratio)

error_ratio.plot.hist()
plt.show()

error_ratio.sort_values()
