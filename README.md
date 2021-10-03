<!-- 
<style>
img[src*='#center'] { 
    display: block;
    margin: auto;
}

.markdown-body h2 {
    border-bottom: 3px solid var(--color-border-muted);
}

</style>

-->

## Data Science Projects

The respository contains jupyter notebook codes of data science/machine learning course work and competitions.

The content lays down a general approach that is used for solving Data Science problem.


## Achievements

- 3rd Rank out of 1702 participants in a competition organized by [Analytics India Magazine](https://analyticsindiamag.com/machinehack-winners-how-these-data-science-enthusiasts-solved-the-predicting-the-costs-of-used-cars-hackathon/) for predicting the price of used cars. The winning code is [here](https://github.com/vasim07/MachineHack/blob/master/UsedCars/FinalSubmission.ipynb).
 
<p align="center">
  <img width="400" height="120" src="https://i.ibb.co/YyDXjrF/Capture.png">
</p>
<p align="center">
  <img width="250" height="150" src="https://www.analyticsindiamag.com/wp-content/uploads/2019/08/machinehack_winners_banner.jpg">
</p>

- 15/1187 participats in a competition hosted by Analyticss India Magazine for predicting price of flight tickets.

- 220/10516 - Overall Analytics Vidhya Rak.

## 1) Problem Statement


brief about ps 

E.g., The loan default ratio of our bank is high and needs to be reduced by x% within the next six months; to achieve the objective, we need to classify our loan applications into high risk, moderate risk, or low-risk profile.

![image of opportunity tree]

The problem statement is accompanied by supporting documents such as 

## 2) Data Science Methodology

**2.1) Data Collection**

A well defined problem 

— — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —  — — — — — —

**2.2) Exploratory Data Analysis**

Exploratory Data Analysis is an approach to summarize and identify important characterstics of the data using visualization and statistical plots. It encourages to test hypothesis, detect relationship, analyze missing values and/or detect outliers.

<p align="center">
  <img width="500" height="300" src="https://i.ibb.co/hgr5TNb/del.png">
</p>
Image source: Capstone project @ IIM Ahmedabad.  

— — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —  — — — — — —

**2.3) PreProcessing**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Specifically, **Feature Engineering** refers to extraction of new features/columns using domain knowledge and **Feature Extraction** refers dimensionality reduction using technique such as PCA, LDA etc.   

— — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —  — — — — — —

**2.4) Split Dataset**

Models are build to work on new data - future data. As such, data is split into three sets train data test data & validation data. In a time series model, the order of data split is preserved.

An appropriate data split assist the modeler to avoid overfitting. **Overfitting** occurs when the trained model is not generalized on unseen data, similarly **Underfitting** occurs when model is not able to learn enough from the training data. 

<p align="center">
  <img width="500" height="300" src="https://i.ibb.co/YhCQHms/datasplit.png">
</p>
Image source: Self's Machine Learning Project.

The benefit of data split can be seen in the above (learning curves) plots. Both plots indicates overfitting. Had data not been split, it would have been difficult to notice this while modeling.

— — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —  — — — — — —

**2.5) Model Training**

Based on the target variable, we can either train a classification or regression model. Model training is an integral part of the entire process, and we can see that it does have various complexity involved. 

Commonly used models

 - Neural Networks (CNN, RNN, GAN, more)
 - Decision Tree-based model (XGboost, LightGBM, Random Forest)
 - Linear model (Linear & Logistic Regression, Linear SVM)

Apart from the above, [here](http://topepo.github.io/caret/available-models.html) we can find more than 200 different models.

— — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —  — — — — — —

**2.6) Hyperparameter tuning**

There are a few methods such as grid search, random search, and bayesian search to identify the optimal hyperparameters of a model.

A practical approach to identify hyperparameter is to use the cross-validation technique. There are various cross-validation techniques such as k-fold cv, Leave one out cv, k-fold stratified cross-validation, etc., which can be selected based on the objective.

<p align="center">
  <img width="200" height="100" src="https://upload.wikimedia.org/wikipedia/commons/c/c7/LOOCV.gif">
</p>


<p align="center"> source: https://en.wikipedia.org</p>

— — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —  — — — — — —

**2.7) Version Control (ML Flow)**

A predictive model design is an iterative process, which involves trying different features, various models and tuning the best hyperparameters. Because of an iterative approach, we need an appropriate version control mechanism to capture version history. 

I prefer using the [ML Flow](https://mlflow.org/) platform from databricks to effectively store and manage model, parameters, evaluation, and artifacts of the model. The simple UI makes it extremely easy to compare performance across different models.

![Image result for mlflow](https://databricks.com/wp-content/uploads/2018/06/mlflow-web-ui.png)
source:- [Databricks](https://databricks.com/blog/2018/06/05/introducing-mlflow-an-open-source-machine-learning-platform.html)

— — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —  — — — — — —

**2.8) Debugging or eXplainable AI**

Significant variables / Feature Importance: Significant variables and feature importance plot provide us the important variables used in the model. With domain expertise, we can match these variables to ensure the stability of the model.

Partial Dependence Plot: The partial dependence plot (short PDP or PD plot) shows the marginal effect one or two features have on the predicted outcome of a machine learning model

Individual Conditional Expectation: Individual Conditional Expectation (ICE) plots display one line per instance that shows how the instance's prediction changes when a feature changes.

SHapley Additive exPlanation (SHAP): The goal of SHAP is to explain the prediction of an instance x by computing the contribution of each feature to the prediction. The SHAP explanation method computes Shapley values from coalitional game theory.

<p align="center">
  <img width="500" height="300" src="https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/boston_dependence_plot.png">
</p>

<p align="center"> source: Python SHAP package</p>

Learning Curves: We can use Learning curves to understand and evaluate the bias variance trade off of the model.

— — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —  — — — — — —

**2.9) Model Evaluation**

Only once we have finalized a model. We evaluate its performance on the test dataset. Our objective is to find out the best model that can be generalized on unseen data. High performance on the test dataset can help us conclude that the model has found the best parameters used for our data.

Even though the modeling building process ends here, however, this does not meet our end objective.

## 3) Deployment & Monitorig

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

**Docker**
In some organizations, the job of a data scientist ends once we have a finalized the model. However, many organizations expect data scientists to provide a production-ready model to the development team.

We can use docker images to build the model pipeline. A docker image is a document with a set of instructions for creating the container. The steps include all relevant information such as the operating system, packages, files, and folders, etc., required to build the same environment for our model.


With all these steps involved, we can notice that building a robust real-time model is a humongous task and requires special skills and a team to keep it up are running.

**References**
1) Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
2) Python Machine Learning
3) [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/)
4) [Machine Learning Mastery Blog](https://machinelearningmastery.com/)
