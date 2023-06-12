# **KAGGLE**.
All or most of projects I have uploaded on Kaggle are gonna be here.
..
 # N1 | Cracking the Code of Accuracy| Unveiling MSE , MAE , RMSE .ipynb 

 
 https://www.kaggle.com/code/darshanprabhu09/mse-mae-rmse-for-the-housing-dataset

I am here to announce and ask for your reviews as I share my First and foremost implementation of Metrics like "Root mean Square error (RMSE)" , "Mean Absolute Error (MAE) " , "Mean Squared Error (MSE)" in Python . I have used the dataset of 5000 peoples whose housing were located also I am gonna share about the statistical as well as the Theory knowledge I gained from this :

RMSE (Root mean squared error)
```
Root mean square error or root mean square deviation is one of the most commonly used measures for evaluating the quality of 
predictions. It shows how far predictions fall from measured true values using Euclidean distance.

To compute RMSE, calculate the residual (difference between prediction and truth) for each data point, compute the norm of 
residual for each data point, compute the mean of residuals and take the square root of that mean.RMSE is commonly used in 
supervised learning applications,as RMSE uses and needs true measurements at each predicted data point.

Root mean square error can be expressed as where N is the number of data points, y(i) is the i-th measurement, and y ̂(i) 
is its corresponding prediction.

Note: RMSE is NOT scale invariant and hence comparison of models using this measure is affected by the scale of the data. 
For this reason, RMSE is commonly used over standardized data.
```
Mean Absolute Error (MAE):
```
In the context of machine learning, absolute error refers to the magnitude of difference between the prediction of an 
observation and the true value of that observation. MAE takes the average of absolute errors for a group of predictions 
and observations as a measurement of the magnitude of errors for the entire group. MAE can also be referred as L1 
(loss function).
```
Mean Squared Error (MSE)
```
:

The average squared difference between the estimated values and the actual value. MSE is a risk function, 
corresponding to the expected value of the squared error loss. The fact that MSE is almost always strictly positive 
(and not zero) is because of randomness or because the estimator does not account for information that could 
produce a more accurate estimate.
```

# N2 |Titans in graphs |Exploring the Time Series : 

https://www.kaggle.com/code/darshanprabhu09/n2-titans-in-graphs-exploring-the-time-series/notebook

Time series analysis is a specific way of analyzing a sequence of data points collected over an interval of time. In time series analysis, analysts record data points at consistent intervals over a set period of time 
rather than just recording the data points intermittently or randomly. However, this type of analysis is not merely the act of collecting data over time. 

What sets time series data apart from other data is that the analysis can show how variables change over time. In other words, time is a crucial variable because it shows how the data adjusts over the course of the 
data points as well as the final results. It provides an additional source of information and a set order of dependencies between the data. 

Time series analysis typically requires a large number of data points to ensure consistency and reliability. An extensive data set ensures you have a representative sample size and that analysis can cut through noisy 
data. It also ensures that any trends or patterns discovered are not outliers and can account for seasonal variance. Additionally, time series data can be used for forecasting—predicting future data based on historical data.


# N3|Webscraping Wonders|Web based data extraction 


https://www.kaggle.com/code/darshanprabhu09/n3-webscraping-wonders-web-based-data-extraction


Web scraping is the process of extracting data from websites through either API requests or the Parsing oh HTML files using BeautifulSoup4 and sometimes even through a HTML parser :D . I hope this helps you as you get the idea of extracing data through the Websites. 



https://www.kaggle.com/code/darshanprabhu09/n8-kernels-unveil-the-future-of-drug-chemistry



# N11 | Feature engineering | PCA | dimensioniality Reduction | PCA : 


Feature engineering is the process where one can make changes in their particular data to be used , By reducing or removing the unwanted data in their dataset this process is sometimes also know as "DIMENSIONALITY REDUCTION". 
 Dimensionality reduction is used to reduce the number of dimensions or features in a particular dataframe so as a result our model may not have to face OVERFITTING of data . There contains various compononents for feature 
 engineering they are as follows : 


1. Principal component analysis (PCA) :
```
      Principal component analysis (PCA) is a popular technique for analyzing large datasets containing a high
      number of dimensions/features per observation, increasing the interpretability of data while preserving
      the maximum amount of information, and enabling the visualization of multidimensional data.


     PCA is a statistical technique for reducing the dimensionality of a dataset. This is accomplished by 
     linearly transforming the data into a new coordinate system where (most of) the variation in the data can be 
     described with fewer dimensions than the initial data. Many studies use the first two principal components in 
     order to plot the data in two dimensions and to visually identify clusters of closely related
     data points. Principal component analysis has applications in many fields such as population 
     genetics, microbiome studies, and atmospheric science.
```
