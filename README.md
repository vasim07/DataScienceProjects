<!-- 
<style>
img[src*='#center'] { 
    display: block;
    margin: auto;
}
</style>

-->

## Machine Learning Projects

The respository contains machine learning competitions/hackathons that I have participated in during the extra time. While such competition's objective is to minimize error or maximize accuracy, these competitions allow us with the flexibility to work on various real-world problems and form multi-dimension thinking in solving data science problems. 

 Here's one competition I won, organized by [Analytics India Magazine](https://analyticsindiamag.com/machinehack-winners-how-these-data-science-enthusiasts-solved-the-predicting-the-costs-of-used-cars-hackathon/) for predicting the price of used cars. The winning code is [here](https://github.com/vasim07/MachineHack/blob/master/UsedCars/FinalSubmission.ipynb).
 
<p align="center">
  <img width="460" height="150" src="https://i.ibb.co/YyDXjrF/Capture.png">
</p>
<p align="center">
  <img width="460" height="300" src="https://www.analyticsindiamag.com/wp-content/uploads/2019/08/machinehack_winners_banner.jpg">
</p>

The following contents lay down the principles that I use for solving a Data Science problem. These principles are inspired by the popular [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) framework.



## Problem Statement

The first step in solving any problems requires a clear definition of the problem statement and [SMART](https://en.wikipedia.org/wiki/SMART_criteria) objectives. 

E.g., The loan default ratio of our bank is high and needs to be reduced by x% within the next six months; to achieve the objective, we need to classify our loan applications into high risk, moderate risk, or low-risk profile.

**Process Mapping**
A process map is a planning and management tool that visually describes the flow of work. Such a visual approach provides a bird's eye view of the business and allows us to see the larger picture of the business flow.

<p align="center">
  <img width="560" height="300" src="https://i.ibb.co/zsqrJN8/test.gif">
</p>

<p align="center"> source: Self (Recruitment process)</p>

A good process map allows us to explain clearly and evaluate the proposed changes and understand the benefits, challenges, and requirements of the proposed structure.

**Performance Measure**
A Performance measure is a numerical value that we want to optimize for the particular model. We should consider appropriate cost activities. 

E.g., What is the cost involved if we reject a low-risk application? Simultaneously what is the cost of a high-risk customer is classified as a low-risk profile.

Apart from the performance measure, we should also define a product success measure.

**Assumptions**
It never hurts to list down your assumptions beforehand while approaching a problem.  If we have a clear understanding of our assumptions, it becomes flexible to switch gears while solving the problem.


## Approach

**Data Collection**

For the competition, we are generally provided with neat and clean data; however, this is not always the case while working on real-world problems. We may need to extract data from multiple sources such as SQL or NO SQL Databases - on-premise, on the cloud, Hadoop, API, or even need to scrape a website.

Based on my experience, for a transactional process, the [Kimball dimensional modeling](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/) approach provides the maximum benefit when it comes to slicing and dicing the data. 

<p align="center">
  <img width="560" height="300" src="https://media.geeksforgeeks.org/wp-content/uploads/20200717001130/Kimball.png">
</p>
<p align="center"> source: www.geeksforgeeks.org</p>

**Exploratory Data Analysis**

An EDA is an essential and integral part of a data science process. It helps us to detect bias, incorrect, incomplete, or outlier records. Additionally, it allows us to test our data distribution assumptions, relationship, and patterns. 

**PreProcessing**

With EDA insights, we get a good idea of what preprocessing needs to be performed on the data. General steps involved are imputing missing data, handle outliers, converting categorical variables.

There are two methods to implement preprocessing steps in the predictive model, either through a pipeline or an intermediate database with preprocessing steps.

**Feature Engineering and Feature Extraction**

All models have a basic assumption, i.e., all the selected features can explain/predict the target variable. Hence, we should aim to choose accurate features during data modeling.

Domain-specific selection:-  Humans are superior in identifying patterns; thus, it is vital to get domain-specific knowledge while selecting features that influence the predicted variable.

Feature Importance Based Selection: Many models such as linear-based or tree-based models provide model-based feature significance and importance. We can use these features to hypothesize our assumptions and design a feature selection process.

Principal Component Analysis (PCA): We can use the PCA technique to extract linear combinations of numerical variables.


<p align="center">
  <img width="600" height="250" src="https://i.ibb.co/wwr2hqh/Capture.png">
</p>

<p align="center"> source: setosa.io</p>


Standardization & Normalization: Some algorithms such as gradient descent and support vector machines boost their speed with standardization and normalization.

$Standardizaion = Z=\dfrac{x-\mu} {\sigma}$    

$Normalization = \dfrac{X-Xmin}{Xmax - Xmin}$

**Train, Test, and Dev Dataset**

To ensure the generalizability of the model, the model's performance must be evaluated on unseen data. One method to achieve this is randomly splitting our dataset into two or three sets, either as train & test set or train, test & validation set.

In a time series model, the order of data inflow is important, as such data is split into periods, e.g., 2017-2018 for the test, 2019 as validation, and 2020 as the test set.

The other approach used for validating a model involves **cross-validation**. In this method, the train dataset is divided into multiple train and validation set. This method can be repeated several times in order to get a more robust model.

<p align="center">
  <img width="400" height="100" src="https://upload.wikimedia.org/wikipedia/commons/c/c7/LOOCV.gif">
</p>


<p align="center"> source: https://en.wikipedia.org/wiki/Cross-validation_(statistics)</p>

## Model Training

Based on the target variable, we can either train a classification or regression model. Model training is an integral part of the entire process, and we can see that it does have various complexity involved. 

Commonly used models

 - Neural Networks (CNN, RNN, GAN, more)
 - Decision Tree-based model (XGboost, LightGBM, Random Forest)
 - Linear model (Linear & Logistic Regression, Linear SVM)

Apart from the above, [here](http://topepo.github.io/caret/available-models.html) we can find more than 200 different models.

**Hyperparameter tuning**

There are a few methods such as grid search, random search, and bayesian search to identify the optimal hyperparameters of a model.

A practical approach to identify hyperparameter is to use the cross-validation technique. There are various cross-validation techniques such as k-fold cv, Leave one out cv, k-fold stratified cross-validation, etc., which can be selected based on the objective.

**Version Control (ML Flow)**

A predictive model design is an iterative process, which involves trying different features, various models and tuning the best hyperparameters. Because of an iterative approach, we need an appropriate version control mechanism to capture version history. 

I prefer using the [ML Flow](https://mlflow.org/) platform from databricks to effectively store and manage model, parameters, evaluation, and artifacts of the model. The simple UI makes it extremely easy to compare performance across different models.

![Image result for mlflow](https://databricks.com/wp-content/uploads/2018/06/mlflow-web-ui.png)
source:- [Databricks](https://databricks.com/blog/2018/06/05/introducing-mlflow-an-open-source-machine-learning-platform.html)

**Debugging ML Models**

Significant variables / Feature Importance: Significant variables and feature importance plot provide us the important variables used in the model. With domain expertise, we can match these variables to ensure the stability of the model.

Partial Dependence Plot: The partial dependence plot (short PDP or PD plot) shows the marginal effect one or two features have on the predicted outcome of a machine learning model

Individual Conditional Expectation: Individual Conditional Expectation (ICE) plots display one line per instance that shows how the instance's prediction changes when a feature changes.

SHapley Additive exPlanation (SHAP): The goal of SHAP is to explain the prediction of an instance x by computing the contribution of each feature to the prediction. The SHAP explanation method computes Shapley values from coalitional game theory.

![](https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_dependence_plot.png =500x300)

source:- [Shap](https://github.com/slundberg/shap)

Learning Curves: We can use Learning curves to understand and evaluate the bias variance trade off of the model.

**Model Evaluation**

Only once we have finalized a model. We evaluate its performance on the test dataset. Our objective is to find out the best model that can be generalized on unseen data. High performance on the test dataset can help us conclude that the model has found the best parameters used for our data.

Even though the modeling building process ends here, however, this does not meet our end objective.

## Deployment, Testing & Monitor

**Docker**
In some organizations, the job of a data scientist ends once we have a finalized the model. However, many organizations expect data scientists to provide a production-ready model to the development team.

We can use docker images to build the model pipeline. A docker image is a document with a set of instructions for creating the container. The steps include all relevant information such as the operating system, packages, files, and folders, etc., required to build the same environment for our model.

**Batch Prediction**
Next, we go into more details about the prediction of models once deployed.

In a situation where we do not need real-time prediction, e.g., predicting employee attrition, we can process our data in batches. 

We set up scheduled jobs to generate predictions in specific intervals such as hourly, daily, or weekly. Such batch predictions are usually part of the ETL process whereby the predictions are stored in databases, preferably a relational database management system. 

Google Big Query, Amazon's Redshift, and Azure Synapse are all data warehouses that can be used to store and process batch predictions. For on premise, we can use any column-oriented relationship database management system such as SQL Server or Oracle SQL.

**Rest API**
In cases where we need real-time prediction, e.g., an Application Programming Interface (API) can be set up. In Python, we can use flask, Django, or fast API to generate API.

A few points to consider while using API.

 - Testing: Ensure that API is tested thoroughly in terms of security, performance, reliability, and maintainability.
 
 - Asynchronization:  Our system should handle each request separately; this becomes extremely important when serving a large number of an API request.
 
 - Task queue: Even with the best infrastructure, systems do go down. Using a task queue, we can be sure that request received during downtime is served once the system is back up and running. 
 
 <p align="center">
  <img width="500" height="300" src="https://miro.medium.com/max/1013/0*0dKAMd8DZkAIQVR3">
</p>

 source:- [Prateek's Blog](https://medium.com/modern-nlp/101-for-serving-ml-models-10217c9f0764)
 
 - CICD: Model deployment is more like software engineering; thus, a DevOps approach such as continuous improvement and continuous deployment ensures that the latest updates are tested and applied as soon as possible.
 
 - Cloud Infrastructure: We can use cloud platforms such as GCP, Azure, or AWS to build the required infrastructure. Features such as compute engine, lambda functions, virtual machines, etc., are easily set up for prototyping and converting to production. Lastly, automatic scaling through a load balancer is extremely helpful to scale up or scale down our API request as per the demand.

With all these steps involved, we can notice that building a robust real-time model is a humongous task and requires special skills and a team to keep it up are running.

**References**
1) Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
2) Python Machine Learning
3) [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
4) [Machine Learning Mastery Blog](https://machinelearningmastery.com/)
