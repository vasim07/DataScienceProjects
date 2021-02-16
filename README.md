
## Machine Learning Projects

The respository contains all machine learning competition/hackathons that I have participated in. Even though the objective of such competition is to minimize error or maximize accuracy, these competitions allow us the flexibility to work on various real world problems and form a multi-dimension thinking in solving data science problems. 

 Here's one competition where I won orginized by [Analytics India Magazine](https://analyticsindiamag.com/machinehack-winners-how-these-data-science-enthusiasts-solved-the-predicting-the-costs-of-used-cars-hackathon/) for predicting price of used cars. The winning code can be found [here](https://github.com/vasim07/MachineHack/blob/master/UsedCars/FinalSubmission.ipynb).

!image

The following contents lays down the principles that I use for solving a Data Science problem. These principles are inspired from the popular [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) framework.



## Problem Statement

The first step in solving any problems requires a clear defination of the problem statement along with [SMART](https://en.wikipedia.org/wiki/SMART_criteria) objectives defined. 

For e.g. The loan default ratio of our bank is high and needs to be reduced by x% within the next 6 months, in order to achieve the objective we need to classify our loan applications into high risk, moderate risk or low risk profile.

**A visual approach**
A visual diagram to the problem provides more clarity to 

!image

Such a visual approach provides us with a good insight on what data would be required to solve a particular problem.

**Product measure**
*Before we design the success criteria of our algoritham, it is important that we identify* 

**Performance Measure**
This the measure that we want to optimize for the particular model. Appropriate cost activity should be taken into consideration. 
For e.g. What is the cost involved if we reject a low risk application, simultaneusly what is the cost if a high risk customer is classified as a low risk profile.

**Assumptions**
It never hurts to list down your assumptions beforehand while approaching a problem.  If we have clear understanding of our assumptions it becomes flexible to swtich gears while solving the problem.


## Approach

**Data Collection**

For a competition we are generally provided with neat and clean data, however, this is not always the case while working on real world problems. We may need to extract data from multiple sources such as SQL or NO SQL Databases - on premise or on cloud, Hadoop, API or even need to scrape a website.

Based on my experience, for a transactional process the [Kimball dimensional modeling](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/dimensional-modeling-techniques/) approach provides the maximum benefit when it comes to slicing and dicing the data. 

!Star schema image

**Exploratory Data Analysis**

An EDA is essential and integral part of a data science process. It helps us to detect bias, incorrect, incomplete or outlier records. Additionally it helps us to test our assumptions of the data distrbution, its relationship and patterns. 

**PreProcessing**

With the insights from EDA, we get good insight about what preprocessing needs to performed on the data. General steps involved are imputing missing data, handle outliers, converting categorical variables.

There are two methods to implement preprocessing steps in predictive model either through pipeline or have a intermediate database with preprocessing steps.

**Feature Engineering and Feature Extraction**

All model have a basic assumption, that all the selected features can explain/predict the target variable. Hence, appropriate feature should be selected while data modeling.

Domian specfic selection :-  Humans are superior in identofying patterns, as such it is important to get domain specific knowledge while selecting features that influences the predicted variable.

Feature Importance Based Selection: Many models such as linear based or tree based models provide model based feature significance and importance. We can use these features to hypotheize our assumption and design a feature selection process.

*Principal Component Analysis (PCA): PCA can be used as a feature extraction technique on numerical variables.*  

!image

*Standardization & Normalization: Some algorithams such as gradient descent and support vector machines boost up their speed with standardization and normalization.*

!equation

**Train, Test and Dev Dataset**

In order to ensure the generalizibility of the model, it is important that the models performance is evaluated on unseen data. One of the method to achieve this is randomly splitting our dataset into two or three sets either as train & test set or train, test & validation set.

In a time series model, the order of data inflow is important, as such data is split into periods for e.g. 2017-2018 for test, 2019 as validation and 2020 as test set.

The other approach used for validating a model invloves **cross validation**. In this method the train dataset is divided into multiple train and validation set. This method can repeated over several time in order to get a more robust model.

!cv iamge

## Model Training

Based on target variable, we can either train a classification or regression model. Even though this is integral part of the entire process, we can see that it does have various complexity involved. 

Commonly used models

 - Neural Networks (CNN, RNN, GANS, more)
 - Decision Tree based model (XGboost, LightGBM, Random Forest)
 - Linear model (Linear & Logistic Regression, Linear SVM)

Apart from above, [here](http://topepo.github.io/caret/available-models.html) we can find more than 200 different models.

**Hyperparamter tuning & CV**

There are a few method such as grid search, random search and bayesian search to identify the optimal hyperparameters of a model.

A pratical approach to identify hyperparameter is to use cross validation techinque. There are various cross validation technique such as k-fold cv, Leave one out cv, k-fold stratified cross validation etc which can be selected based on the objective.

**Version Control (ML Flow)**

A predictive model design is an iterative process, which involves trying different features, various models and tuning the best hyperparamters. Because of an iterative approach we need a appropriate version control mechanism to capture version history. 

I prefer using the [ML Flow](https://mlflow.org/) platform from databricks to effectively store and manage model, parameters, evaluation and artifacts of the model. The simple UI makes it extremely easy to compare performance across different models.

!Ui image


**Debugging ML Models**

Significant variables / Feature Importance: Significant varaibles and feature importance plot provides us the important variables used in the model. These variables can be matched with domain expertize to ensure stability of the model.

Partial Dependence Plot: The partial dependence plot (short PDP or PD plot) shows the marginal effect one or two features have on the predicted outcome of a machine learning model

Individual Conditional Expectation: Individual Conditional Expectation (ICE) plots display one line per instance that shows how the instance's prediction changes when a feature changes.

SHapley Additive exPlanation (SHAP): The goal of SHAP is to explain the prediction of an instance x by computing the contribution of each feature to the prediction. The SHAP explanation method computes Shapley values from coalitional game theory.

!SHAP iamge

Learning Curves

**Model Evaluation**

Only once we have finalized a model. We evalute its performance on the test dataset. Our objective is to find out the best model that can be generalized on unseen data. A high performance on the test dataset can help us to conclude that the model has found out the best parameters used for our data.

Even though the modeling building process ends here. However, this does not end our objective.

## Deployment, Testing & Monitor

**Dokcer**
In some organisation, the job of a data scientist ends once we have a finalized the model. However, many organisation expecct data scientist to provide a production ready model to the development team.

We can use docker images to build the model pipeline. A docker image is a document with set of instructions for creating container. The steps include all relevant information such as operating system, packages, files and folders, etc required to build the exact environment for our model.

**Batch Prediction**
Next, we go into more details about the prediction of models once deployed.

In situation were we do not need real time prediction for e.g. predicting employee attrition, we can process our data in batches. 

We setup scheduled jobs to generate prediction in specific interval such as hourly, daily or weekly. Such batch predictions are usually part of ETL process whereby the predictions are stored in databases, preferably a relationship database management system. 

Google Big Query, Amazon's Redshift and Azure Synapse are all data warehouse that can be used to store and process batch predictions. For on premise we can use any column oriented relationship database management system such as SQL server or Oracle SQL.

**Rest API**
In cases where we need real time prediction for e.g. an Application Programming Interface (API) can be setup. In python we can use flask, django or fastapi to generate api.

A few points to be considered while using API.

 - Testing: Ensure that API is tested throughly in terms of security, performace, reliability and maintainability.
 
 - Asyncronization:  Each request should be handled seperately, this becomes extremely important when serving a large number of API request.
 
 - Task queue: Even with best infrastructure system do go down. By using a task queue we can be sure that request received during downtime are served once the system is back up and running. 
 
 - CICD: Model deployemnet is more as a software engennering, as such a devops approach such as continous imporevment and continous deployement ensures that the latest updates are tested and applied as soon as possible.
 
 - Cloud Infrastructure: We can use cloud platform such as GCP, Azure or AWS to build the required infrastruture. Features such as compute engine, lambda functions, virtual machines etc are easily setup for prototyping and converting to production. Lastly automatic scaling through load balancer are extremely helpful to scale up or scale down our API request as per the demand.

With all these steps involved we can notice that buidling a robust real time model is humongous task and requries special skills and team to keep it up are running.

**References**
1) *Hands on Scikit leran and Tensorflow*
2) *Python Machine Learning*
3) [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
4) [Machine Learning Mastery Blog](https://machinelearningmastery.com/)