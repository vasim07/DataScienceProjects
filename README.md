## WORK IN PROGRESS

The following document is work in progress.

## Machine Learning Projects

The respository contains all machine learning competition/hackathons that I have participated in. Even though the objective of such competition is to minimize error or maximize accuracy, these competitions allow us the flexibility to work on various real world problems and form a multi-dimension thinking in solving data science problems. 

Highlight - I would like to share about a competition that I won orginized by [Analytics India Magazine](https://analyticsindiamag.com/machinehack-winners-how-these-data-science-enthusiasts-solved-the-predicting-the-costs-of-used-cars-hackathon/) for predicting price of used cars. The winning code can be found [here](https://github.com/vasim07/MachineHack/blob/master/UsedCars/FinalSubmission.ipynb).

The following contents lays down the principles that I use for solving a Data Science problem. These principles are inspired from the popular [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) framework.



## Problem Statement

The first step in solving any problems requires a clear defination of the problem statement along with [SMART](https://en.wikipedia.org/wiki/SMART_criteria) objectives defined. 

For e.g. The loan default ratio of our bank is high and needs to be reduced by x% within the next 6 months, in order to achieve the objective we need to classify our loan applications into high risk, moderate risk or low risk profile.

**A visual approach**
A visual diagram to the problem provides more clarity to 

Such a visual approach provides us with a good insight on what data would be required to solve a particular problem.

**Product measure**
Before we design the success criteria of our algoritham, it is important that we identify 

**Performance Measure**
This the measure that we want to optimize for the particular model. Appropriate cost activity should be taken into consideration. 
For e.g. What is the cost involved if we reject a low risk application, simultaneusly what is the cost if a high risk customer is classified as a low risk profile.

**Assumptions**
It never hurts to list down your assumptions beforehand while approaching a problem.  If we have clear understanding of our assumptions it becomes flexible to swtich gears while solving the problem.


## Approach

**Data Collection**

For a competition we are generally provided with neat and clean data, however, this is not always the case while working on real world problems. We may need to extract data from multiple sources such as SQL or NO SQL Databases - on premise or on cloud, Hadoop, API or even need to scrape a website.

Based on my experience, for a transactional process the [Kimball dimensional modeling](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/) approach provides the maximum benefit when it comes to slicing and dicing the data. 

**Exploratory Data Analysis**

An EDA is essential and integral part of a data science process. It helps us to detect bias, incorrect, incomplete or outlier records. Additionally it helps us to test our assumptions of the data distrbution, its relationship and patterns. 

**PreProcessing**

With the insights from EDA, we get a good insight about what preprocessing needs to performed on the data. General steps involved are imputing missing data, handle outliers, converting categorical variables.

There are two methods to implement a pipeline in model either through pipeline or have a database which performs the preprocessing steps.

**Feature Engineering and Feature Extraction**

All model have a basic assumption, that all the selected features can explain/predict the target variable. Hence, appropriate feature should be selected while data modeling.

Domian specfic selection :-  Humans are superior in identofying patterns, as such it is important to get domain specific knowledge while selecting features that influences the predicted variable.

Feature Importance Based Selection: Many models such as linear based or tree based models provide model based feature significance and importance. We can use these features to hypotheize our assumption and design a feature selection process.

*Principal Component Analysis (PCA): PCA can be used as a feature extraction technique on numerical variables.*  

*Standardization & Normalization: Some algorithams such as gradient descent and support vector machines boost up their speed with standardization and normalization.*

**Train, Test and Dev Dataset**

In order to ensure the generalizibility of the model, it is important that the models performance is evaluated on unseen data. One of the method to achieve this is randomly splitting our dataset into two or three sets either as train & test set or train, test & validation set.

In a time series model, the order of data inflow is important, as such data is split into periods for e.g. 2017-2018 for test, 2019 as validation and 2020 as test set.

The other approach used for validating the dataset invloves cross validation. In this method the train dataset is divided into multiple train and validation set. This can repeated over several time in order to get a more robust model.

### Model Training

Based on target variable, we can either train a classification or regression model. Even though this is integral part of the entire process, we can see that it does have various complexity involved. The reason for this is the appropriate process followed.

### Hyperparamter tuning & CV

There are a few method such as grid search, random search and even bayesian search to identify the optimal hyperparameters of a model.

A pratical approach to identify hyperparameter is to use cross validation techinque. Their are various cross validation technique such as k-fold cv, Leave one out cv, k-fold stratified cross validation etc.

### Version Control (ML Flow)

A predictive model design is an iterative process, which involves trying different features, various models and tuning the best hyperparamters. Because of an iterative approach we need a appropriate version control mechanism to capture version history. I prefer using the ML Flow program from databricks to effectively store and manage model, parameters and evaluation history.

### Model Evaluation

During the process of modeling, we need to evaluate the performance of the model. 

### Debugging ML Models

Learning Curves

Model Complexity

Feature Importance

Partial Dependence Plot

Individual Conditional Expectation

SHapley Additive exPlanation (SHAP)

Fairness

## Deployment, Testing & Monitor

Batch Prediction

Rest API

Version Control

Dokcer & Kubernetes





