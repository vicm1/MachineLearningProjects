import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


csv_file_path = r'MachineLearningProjects\MachineLearningModel\teams.csv' # store the file path of the csv in a variable for easier access
teams = pd.read_csv(csv_file_path)

teams = teams[["team","country","year","athletes","age","prev_medals","medals"]] # make data with new columns that show only data needed, removing unnecessary columns. 


sns.lmplot(x="athletes", y = "medals", data = teams, fit_reg = True, ci= None) # plot graph that shows linear relationship of number of athletes that a country enters to the olympics more medals earned

sns.lmplot(x="age", y = "medals", data = teams, fit_reg = True, ci= None) # plot graphs that shows how many medals won by age, seems that age doesn't come into large play when winning medals

teams.plot.hist(y="medals")
plt.show() # show the plot
