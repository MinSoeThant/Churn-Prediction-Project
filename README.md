# Bank Customer Churn Prediction
### Introduction
- To identify and select the features that affects customer churn
- To build a suitable model with high predictability on customer churn for deployment
### About Dataset
#### Features:
- Surname: Surname
- CreditScore: Credit score
- Geography: Country (Germany / France / Spain)
- Gender: Gender (Female / Male)
- Age: Age
- Tenure: How many years of customer
- Balance: Balance
- NumOfProducts: Bank product used
- HasCrCard: Credit card status (0 = No, 1 = Yes)
- IsActiveMember: Active membership status (0 = No, 1 = Yes)
- EstimatedSalary: Estimated salary
- Exited: Churn or Not Churn (0 = No, 1 = Yes)


## Thought Process 
I will be sharing with you the thought process and methods that I have chosen to execute the project.

Univariate, bivariate and multivariate analysis are conducted to perform exploratory data analysis.
Univariate Analysis
- Firstly, mean, median, variance and standard deviation value is analysed.
- Secondly, I analysed the data to find out whether it follows a normal distribution using kde and histogram plot.
- Kde and histogram plot is used to show how the data plotted are equally distributed and fits a bell curve.
- As shown on the plot, the graph is not normally distributed. Since the p-value is less than 0.05, the null hypothesis that the data are normally distributed is rejected. Therefore, I had to figure out skewness and kurtosis of the graph since the plot is not normally distributed.
- I used the quantile-quantile plot to assess the normal distribution of the graph. This is to validate the hypothesis testing that shows the data is not normally distributed.
Box plots are used to identify outliers in the data.

Bivariate Analysis
- Numerical-Numerical variables, Numerical-Categorical variables, Categorical-Categorical variables is compared for bivariate analysis
- Since I discovered that there is little to no linear correlation between numerical-numerical variables with Pearson correlation cofficient, I decided to use the Phi_K correlation coefficient to find the non-linear correlation between numerical-numerical variables.

Multivariable Analysis
- Factor analysis is used to perform multivariable analysis.
- However, when I used the Kaiser-Meyer-Olkin (KMO) test to determine the adequancy for factor analysis, KMO value is less than 0.6 therefore it is inadequate and factor analysis is not preferred.

Modelling
- Data will be cleaned by one-hot encoding, binning and removing missing data if there is any.
- the data sample will be normalised
- Features that are not relevant or may affect the accuracy for prediction are removed.
- K means clustering is not used as a model because the clusters are in different shapes.
- Features selection shows that age, credit score, number of products and balance are the important features for prediction and accuracy.

Saving Model for deployment
- A pipeline between the standard scaler and decision tree classifier model is used to prevent data leak before pickling the file for model deployment.

Model Deployment
- I have used Streamlit as a mean to deploy the model.
- I have difficulty in deploying model to Streamlit Cloud due to the errors that occurred in it. Therefore, I have chosen to deploy the model on the local host cmd terminal with a screen recording as an alternative.
- I have used the data from csv file into the prediction model to show that prediction model is working correctly. 

Model Deployment Video
- Link: https://clipchamp.com/watch/kuMn13qORag
