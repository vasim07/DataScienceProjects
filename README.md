
## Machine Learning Projects

The respository contains all machine learning competition/hackathons that I have participated in. Even though the objective of such competition is to minimize error or maximize accuracy, these competitions allow us the flexibility to work on various real world problems and form a multi-dimension thinking in solving data science problems. 

Since you are here, I would like to share about a competition that I won orginized by [Analytics India Magazine](https://analyticsindiamag.com/machinehack-winners-how-these-data-science-enthusiasts-solved-the-predicting-the-costs-of-used-cars-hackathon/) for predicting price of used cars. The winning code can be found [here](https://github.com/vasim07/MachineHack/blob/master/UsedCars/FinalSubmission.ipynb).

The following contents lays down the principles that I generally use for solving a Data Science problem.



## Problem Statement

The first step in solving any problems requires a clear defination of the problem statement along with [SMART](https://en.wikipedia.org/wiki/SMART_criteria) objectives defined. 

For e.g. The loan default ratio of our bank is high and needs to be reduced by x% within the next 6 months, in order to achieve the objective we need to classify our loan applications into high risk, moderate risk or low risk profile.

**A visual approach**
A visual diagram to the problem provides more clarity to 

**Product measure**
Before we design the success criteria of our algoritham, it is important that we identify

**Performance Measure**
This the measure that we want to optimize for the particular model. Appropriate cost activity should be taken into consideration. 
For e.g. What is the cost involved if we reject a low risk application, similarly what is the cost if a high risk customer is classified as a low risk profile.

**Assumptions**
It never hurts to have your assumptions beforehand while approaching a problem.  If we have clear understanding of our assumptions it is agile to swtich gears while solving the problem.


## Approach

**Data Collection**

For a competition we are generally provided with neat and clean data, however, this is not always the case while working on real world problems. We may need to extract data from multiple sources such as SQL or NO SQL Databases - on premise or on cloud, Hadoop, API or even need to scrape a website.

Based on my experience, for a transactional process the [Kimball dimensional modeling](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/) approach provides the maximum benefit when it comes to slicing and dicing the data. 

**Exploratory Data Analysis**

Detect Outliers, Bias, check assumptions, missing values, capped values, infer relationship

Univariate

Bivariate

Map

**PreProcessing**

Impute Missing Data

Handle Outliers

Handle Categorical variables 

Standardization & Normalization

**Feature Engineering and Feature Extraction**

Domian specfic Selection

Feature Importance Based Selection

PCA

Factor Analysis

**Train, Test and Dev Dataset**

Train Set

Dev Set

Test Set

### Model Training

Supervised

Unsupervised

Version Control (ML Flow)

### Hyperparamter tuning & CV

Grid Search

Random Search

Bayesian search

### Model Debug

Learning Curves

Model Complexity

Feature Importance

Partial Dependence Plot

Individual Conditional Expectation

SHapley Additive exPlanation (SHAP)

Lime

Fairness

## Deployment, Testing & Monitor

Batch Prediction

Rest API

Version Control

Dokcer & Kubernetes





