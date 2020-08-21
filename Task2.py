# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np


# reading a csv file using pandas library
task=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
task.columns

plt.plot(task.Hours,task.Scores,"bo");plt.xlabel("Hours");plt.ylabel("Scores")

task.Scores.corr(task.Hours) # # correlation value between X and Y
np.corrcoef(task.Hours,task.Scores)

# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf
model=smf.ols("Scores~Hours",data=task).fit()
model
# For getting coefficients of the varibles used in equation
model.params
# P-values for the variables and R-squared value for prepared model
model.summary()

model.conf_int(0.05) # 95% confidence interval
pred = model.predict(task.iloc[:,0]) # Predicted values of AT using the model
rmse1 = np.sqrt(np.mean((pred-task.Scores)**2))
rmse1


task["SQSR"] = task.Hours*task.Hours
task.SQSR
model_quad = smf.ols("Scores~Hours+ task.SQSR",data=task).fit()
model_quad.params
model_quad.summary()
model_quad.conf_int(0.05) # 95% confidence interval
pred1 = model_quad.predict(task.iloc[:,0]) # Predicted values of AT using the model
rmse2 = np.sqrt(np.mean((pred1-task.Scores)**2))
rmse2


model_log = smf.ols("Scores~np.log(Hours)",data=task).fit()
model_log.params
model_log.summary()
model_quad.conf_int(0.05) # 95% confidence interval
pred2 = model_log.predict(task.iloc[:,0]) # Predicted values of AT using the model
rmse3 = np.sqrt(np.mean((pred2-task.Scores)**2))
rmse3

model_S = smf.ols("Scores~Hours**3 + task.SQSR ",data=task).fit()
pred4 = model_S.predict(task.iloc[:,0]) # Predicted values of AT using the model
rmse4 = np.sqrt(np.mean((pred4-task.Scores)**2))
rmse4
model_S.predict(float(9.25))


model_S = smf.ols("Scores~Hours**3 + task.SQSR ",data=task).fit()
pred4 = model_S.predict(task.iloc[:,0]) # Predicted values of AT using the model
rmse4 = np.sqrt(np.mean((pred4-task.Scores)**2))
rmse4
model_S.predict(float(9.25))


###################################
#rmse2 is having least square
#Selected model : model_quad = smf.ols("Scores~Hours+ task.SQSR",data=task).fit()
#Intercept    3.091536 , Hours        9.475925 ,task.SQSR    0.028652
#For 9.25 hours the predicted score is 90.9998








