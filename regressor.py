"""
    author : Plaban Nayak
"""
from logging import lastResort
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("Advertising.csv")
print(df.head())

df['total_amount_spent'] = df['TV'] + df['newspaper'] + df['radio']
print(df.head())


sns.scatterplot(x='total_amount_spent',y='sales',data=df)
plt.title('Total amount spent on advertising VS Sales')
plt.show()
# reg plot draws the line of best fit
sns.regplot(data=df,x='total_amount_spent',y='sales',color='red')
plt.show()

x = df['total_amount_spent']
y = df['sales']
"""
# the polyfit function gives the values of slope and intercept for data points
# [0.04868788==> theta1(slope) 4.24302822 ==> theta0(intercept)]
"""
print(np.polyfit(x,y,deg=1))# applicable to linear regression to get intercept and slope

print(np.polyfit(x,y,deg=2))
#[8.21901022e-06 4.53486875e-02 4.51143622e+00]
slope,intercept = np.polyfit(x,y,deg=1)
potential_spend = np.linspace(0,500,100)
predicted_sales = 0.04868788 * potential_spend + 4.24302822 
sns.scatterplot(x='total_amount_spent',y='sales',data=df)
#draw the regression line
plt.plot(potential_spend,predicted_sales,color='red')
plt.show()
#here regplot and polyfit results ploting attain the same results

spend =200
predicted_sales = slope * spend +  intercept
print(f"Predcted Sales :{predicted_sales}")
#
x,y,z,w = np.polyfit(x,y,deg=3) # ploynomial equation degree =3 
pot_spend = np.linspace(0,500,100)
pred_pol = x * pot_spend **3 + y * pot_spend**2 + z * pot_spend + w
sns.scatterplot(x='total_amount_spent',y='sales',data=df)
#
#draw the regression line
#
plt.plot(pot_spend,pred_pol,color='red')
plt.title("Polynomial pred deg 3")
plt.show()

fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(14,6))
axes[0].plot(df['TV'],df['sales'],'o')
axes[0].set_ylabel('sales')
axes[0].set_title('amount spent on TV')

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].set_ylabel('sales')
axes[1].set_title('amount spent on radio')

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].set_ylabel('sales')
axes[2].set_title('amount spent on  newspaper')
plt.show()

# pair plot
sns.pairplot(df)
plt.show()
# features and label segregation
X = df.drop(['sales','total_amount_spent'],axis=1)
y = df[['sales']]
# train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.33,random_state=101)
#build model
lr = LinearRegression()
# train modelgit branch
lr.fit(X_train,y_train)
#evaluate the model
y_pred = lr.predict(X_test)
# plot predictions
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


# 
print(f'Mean absoute error: {mean_absolute_error(y_test,y_pred)}')
print(f'Mean squared error: {mean_squared_error(y_test,y_pred)}')
print(f'Root Mean squared error: {np.sqrt(mean_squared_error(y_test,y_pred))}')
print(f'Root Mean squared error: {np.sqrt(r2_score(y_test,y_pred))}')
#
"""
if RMSE > MAE  ==> possible changes of Outliers
if RMSE approximates to  MAE then results are good and no outliers present
"""

# visualize predictions
sns.regplot(x=X_test['TV'],y=y_test,color='red')
plt.show()
sns.regplot(x=X_test['TV'],y=y_pred,color='red')
plt.show()
'''
# Residual plot helps to visualize if linear regression will be applicable to the dataset
# calculate residuals
1. using seaborn
'''

sns.residplot(x='TV', y='sales', data=df)
plt.show()
orig = y_test
pred = y_pred
test_residuals = orig - pred
print(y_test)
print(test_residuals)


sns.scatterplot(x=y_test['sales'],y=test_residuals['sales'])
plt.axhline(y=0,color='red',ls='--')
plt.show()

sns.distplot(test_residuals.values.tolist(),bins=25,kde=True)
plt.show()

'''
viusalize the relationship between features and preicted and orinal sales values
'''
yhat = lr.predict(X)
fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(14,6))
axes[0].plot(df['TV'],df['sales'],'o',label='Sales')
axes[0].plot(df['TV'],yhat,'o',color='green',label='Predicted Sales')
axes[0].set_ylabel('sales')
axes[0].set_title('amount spent on TV')

axes[1].plot(df['radio'],df['sales'],'o')
axes[1].plot(df['radio'],yhat,'o',color='green')
axes[1].set_ylabel('sales')
axes[1].set_title('amount spent on radio')

axes[2].plot(df['newspaper'],df['sales'],'o')
axes[2].plot(df['newspaper'],yhat,'o',color='green')
axes[2].set_ylabel('sales')
axes[2].set_title('amount spent on  newspaper')

fig.legend(axes, labels=['sales','predicted_sales'],loc="upper right")
plt.show()