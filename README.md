# **KAGGLE**.
All or most of projects I have uploaded on Kaggle are gonna be here.
..
# N1 | Cracking the Code of Accuracy| Unveiling MSE , MAE , RMSE .ipynb 

 
 https://www.kaggle.com/code/darshanprabhu09/mse-mae-rmse-for-the-housing-dataset

I am here to announce and ask for your reviews as I share my First and foremost implementation of Metrics like "Root mean Square error (RMSE)" , "Mean Absolute Error (MAE) " , "Mean Squared Error (MSE)" in Python . I have used the dataset of 5000 peoples whose housing were located also I am gonna share about the statistical as well as the Theory knowledge I gained from this :

RMSE (Root mean squared error)
```
Root mean square error or root mean square deviation is one of the most commonly used measures 
for evaluating the quality of predictions. It shows how far predictions fall from measured
true values using Euclidean distance.

To compute RMSE, calculate the residual (difference between prediction and truth) for each
data point, compute the norm of residual for each data point, compute the mean of residuals 
and take the square root of that mean.RMSE is commonly used in supervised learning
applications,as RMSE uses and needs true measurements at each predicted data point.

Root mean square error can be expressed as where N is the number of data points, y(i) 
is the i-th measurement, and y ̂(i) .Is its corresponding prediction.

Note: RMSE is NOT scale invariant and hence comparison of models using this 
measure is affected by the scale of the data. For this reason, RMSE is
commonly used over standardized data.
```
Mean Absolute Error (MAE):
```
In the context of machine learning, absolute error refers to the magnitude
of difference between the prediction of an observation and the true value 
of that observation. MAE takes the average of absolute errors for a group 
of predictions and observations as a measurement of the magnitude of 
errors for the entire group. MAE can also be referred as L1 
(loss function).

```
Mean Squared Error (MSE) :
```
:

The average squared difference between the estimated values and the actual
value. MSE is a risk function, corresponding to the expected value of the 
squared error loss. The fact that MSE is almost always strictly positive 
(and not zero) is because of randomness or because the estimator does not
account for information that could produce a more accurate estimate.
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


Web scraping is the process of extracting data from websites through either API requests or the Parsing oh HTML files using BeautifulSoup4 and sometimes even through a HTML parser :D . I hope this helps you as you get
the idea of extracing data through the Websites. 

- WEB SCRAPING : 

```
Before we jump into the code lets know a little about Webscraping and how 
it can be useful for the purpose of data Analysis. Being in the Field of 
Data analysts , I always wondered whether we needed to have data in our 
local PC to form operations like alterations , Removing inconsistencies, 
Gaps and Redundancies in our data , Happens to be that there is a 
concept known as "WEBSCRAPING" . Do go through the small Snippet of 
code I have created using Python version 3.9.2 .

```

To know more about this module refer : https://pypi.org/project/beautifulsoup4/


-  Libraries and modules and how to install them:

```
1.) beautifulsoup4 4.12.2 : Beautiful Soup is a library that makes it
easy to scrape information from web pages.It sits atop an HTML or 
XML parser, providing Pythonic idioms for iterating, searching,
and modifying the parse tree. It helps us basically to read the HTML
files we have on any particular websites. To install this library
use the syntax below :

```


```

 pip install beautifulsoup4

```


https://www.kaggle.com/code/darshanprabhu09/n8-kernels-unveil-the-future-of-drug-chemistry


# N8|Kernels Unveil the Future of Drug | CHEMISTRY. | Version 4 : 

https://www.kaggle.com/code/darshanprabhu09/n8-kernels-unveil-the-future-of-drug-chemistry

 I have done EDA on the data and then after doing so I created a Linear Regression and a Random forest model to check the accuracy 
 of our model and then chose the final model to be used , I will really appreciate it if you guys took out your spare time to check out
 my work and share your reviews or some suggestions meant to be added on the same .
 
 
 About the dataset : 
 
 The given data appears to be a tabular representation with several columns. Here is a description of each column:

MolLogP: This column represents the calculated logarithm of the partition coefficient (LogP) for the chemical compounds. LogP is a measure of the compound's lipophilicity or hydrophobicity, indicating its tendency to dissolve in oil or water.

MolWt: This column represents the molecular weight of the compounds. Molecular weight is the sum of the atomic weights of all atoms in a molecule and is an important property in determining the physical and chemical characteristics of a compound.

NumRotatableBonds: This column indicates the number of rotatable bonds in the chemical compounds. Rotatable bonds refer to single bonds that can rotate freely without breaking, and they play a role in the compound's flexibility and conformational changes.

AromaticProportion: This column represents the proportion of aromatic atoms present in the compounds. Aromaticity is a property associated with compounds containing conjugated ring systems, such as benzene rings.

logS: This column represents the logarithm of the solubility values for the compounds. Solubility is a measure of how well a compound dissolves in a given solvent.

**Label**: This column appears to provide ranges or intervals for the solubility values (logS) or molecular weights (MolWt) in the dataset.

**Count**: This column represents the count or frequency of compounds falling within the respective range or interval specified by the label.



# N9|Feature Engineering|PCA|DIMENSION-REDUCTION : 



Feature engineering is the process where one can make changes in their particular data to be used , By reducing or removing the unwanted data in their dataset this process is sometimes also know as "DIMENSIONALITY REDUCTION". 
 Dimensionality reduction is used to reduce the number of dimensions or features in a particular dataframe so as a result our model may not have to face OVERFITTING of data . There contains various compononents for feature 
 engineering they are as follows : 


1. Principal component analysis (PCA) :
```
      Principal component analysis (PCA) is a popular technique for analyzing large datasets containing a high
      number of dimensions/features per observation, increasing the interpretability of data while preserving
      the maximum amount of information, and enabling the visualization of multidimensional data.


     PCA is a statistical technique for reducing the dimensionality of a dataset. This is accomplished by 
     linearly transforming the data into a new coordinate system where (most of) the variation in the data 
     can be described with fewer dimensions than the initial data. Many studies use the first two principal
     components in order to plot the data in two dimensions and to visually identify clusters of closely
     related data points. Principal component analysis has applications in many fields such as population 
     genetics, microbiome studies, and atmospheric science.
```


2. Dimensionality Reduction :

```
Dimensionality reduction is a technique used to reduce the number of features in a dataset while retaining
as much of the important information as possible. In other words, it is a process of transforming 
high-dimensional data into a lower-dimensional space that still preserves the essence of the original data.

In machine learning, high-dimensional data refers to data with a large number of features or variables.
The curse of dimensionality is a common problem in machine learning, where the performance of the 
model deteriorates as the number of features increases. This is because the complexity of the model 
increases with the number of features, and it becomes more difficult to find a good solution. In 
addition, high-dimensional data can also lead to overfitting, where the model fits the training
data too closely and does not generalize well to new data.

Dimensionality reduction can help to mitigate these problems by reducing the complexity of the model and 
improving its generalization performance. There are two main approaches to dimensionality reduction:
feature selection and feature extraction.

```

3. Feature Selection:

```
Feature selection involves selecting a subset of the original features that are most relevant to the
problem at hand. The goal is to reduce the dimensionality of the dataset while retaining the most 
important features. There are several methods for feature selection, including filter methods,
wrapper methods, and embedded methods. Filter methods rank the features based on their relevance 
to the target variable, wrapper methods use the model performance as the criteria for selecting 
features, and embedded methods combine feature selection with the model training process.

```


4. Feature Extraction:

```
Feature extraction involves creating new features by combining or transforming the original 
features. The goal is to create a set of features that captures the essence of the original 
data in a lower-dimensional space. There are several methods for feature extraction, including 
principal component analysis (PCA), linear discriminant analysis (LDA), and t-distributed 
stochastic neighbor embedding (t-SNE). PCA is a popular technique that projects the original
features onto a lower-dimensional space while preserving as much of the variance as possible.

```

5. Principal Component Analysis

![image](https://github.com/Darshan0902/KAGGLE./assets/77969007/5581e55a-f44d-4369-8a38-516e841df775)


```
This method was introduced by Karl Pearson. It works on the condition that while the data
in a higher dimensional space is mapped to data in a lower dimension space, the variance 
of the data in the lower dimensional space should be maximum.



It involves the following steps:

Construct the covariance matrix of the data.
Compute the eigenvectors of this matrix.
Eigenvectors corresponding to the largest eigenvalues are used to reconstruct a large
fraction of variance of the original data.Hence, we are left with a lesser number of 
eigenvectors, and there might have been some data loss in the process. But, the most 
important variances should be retained by the remaining eigenvectors. 
```


# N11 | Feature engineering | PCA | dimensioniality Reduction | PCA : 

https://www.kaggle.com/code/darshanprabhu09/n11-feature-engineering-pca-dimensions


Feature engineering is the process where one can make changes in their particular data to be used , By reducing or removing the unwanted data in their dataset this process is sometimes also know as "DIMENSIONALITY REDUCTION". 
 Dimensionality reduction is used to reduce the number of dimensions or features in a particular dataframe so as a result our model may not have to face OVERFITTING of data . There contains various compononents for feature 
 engineering they are as follows : 


1. Principal component analysis (PCA) :
```
      Principal component analysis (PCA) is a popular technique for analyzing large 
      datasets containing a high number of dimensions/features per observation, 
      increasing the interpretability of data while preserving the maximum amount 
      of information, and enabling the visualization of multidimensional data.


     PCA is a statistical technique for reducing the dimensionality of a dataset. 
     This is accomplished by linearly transforming the data into a new coordinate 
     system where (most of) the variation in the data can be described with fewer 
     dimensions than the initial data. Many studies use the first two principal
     components in order to plot the data in two dimensions and to visually 
     identify clusters of closely related data points. Principal component
     analysis has applications in many fields such as population 
     genetics, microbiome studies, and atmospheric science.
     
```

2. Dimensionality Reduction :

```
Dimensionality reduction is a technique used to reduce the number of features
in a dataset while retaining as much of the important information as possible. 
In other words, it is a process of transforming high-dimensional data into a
lower-dimensional space that still preserves the 
essence of the original data.

In machine learning, high-dimensional data refers to data with a large number 
of features or variables. The curse of dimensionality is a common problem in
machine learning, where the performance of the model deteriorates as the 
number of features increases. This is because the complexity of the model 
increases with the number of features, and it becomes more difficult to find 
a good solution. In addition, high-dimensional datacan also lead to 
overfitting, where the model fits the training data too closely 
and does not generalize well to new data.

Dimensionality reduction can help to mitigate these problems by reducing 
the complexity of the model and improving its generalization performance. 
There are two main approaches to dimensionality reduction: feature 
selection and feature extraction.

```

3. Feature Selection:

```
Feature selection involves selecting a subset of the original features
that are most relevant to the problem at hand. The goal is to reduce
the dimensionality of the dataset while retaining the most important
features. There are several methods for feature selection, including
filter methods, wrapper methods, and embedded methods. Filter methods
rank the features based on their relevance to the target variable, 
wrapper methods use the model performance as the criteria for selecting 
features, and embedded methods combine feature selection with the model
training process.

```



4. Feature Extraction:

```
Feature extraction involves creating new features by combining or transforming
the original features. The goal is to create a set of features that captures 
the essence of the original data in a lower-dimensional space. There are 
several methods for feature extraction, including principal component analysis 
(PCA), linear discriminant analysis (LDA), and t-distributed stochastic neighbor 
embedding (t-SNE). PCA is a popular technique that projects the original
features onto a lower-dimensional space while preserving as much of the 
variance as possible.

```

5. Principal Component Analysis

![image](https://github.com/Darshan0902/KAGGLE./assets/77969007/5581e55a-f44d-4369-8a38-516e841df775)


```
This method was introduced by Karl Pearson. It works on the condition
that while the data in a higher dimensional space is mapped to data in 
a lower dimension space, the variance of the data in the lower 
dimensional space should be maximum.



It involves the following steps:

Construct the covariance matrix of the data.
Compute the eigenvectors of this matrix.
Eigenvectors corresponding to the largest eigenvalues are used to 
reconstruct a large fraction of variance of the original data.Hence, 
we are left with a lesser number of eigenvectors, and there might
have been some data loss in the process. But, the most 
important variances should be retained by the remaining eigenvectors.
