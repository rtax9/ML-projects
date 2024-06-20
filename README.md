# Machine Learning Projects

I worked on these projects as part of the __IBM Machine Learning Professional Certification__. 

__1. Deep Learning Project : GAN for generating handwritten images using MNIST Data__

    - _Data Source:_
    MNIST dataset 

    - _Objective:_
    To train a deep convolutional generative adversarian network (__DCGAN__)

    - _Description:_ 


__2. Supervised Learning- Classification : FDA_Drug_Classification__

    - _Data Source:_
    https://www.kaggle.com/datasets/maheshdadhich/us-healthcare-data?datasetId=7684&sortBy=dateRun&tab=profile 

    - _Objective:_
    The objective of this project is to predict the catergory of any given drug in terms of one of the following

        - OTC (over the counter)
        - Prescription
        - Allergenic
          or

        - Other
  
    This is a __classification problem__ and we shall implement several different types of classification models to make the above prediction.


    - _Description:_ 

    This dataset is interesting for couple of its intriguing characteristics, namely, __ALL the features of this dataset are categorical!__. That is, there are no numerical values in this dataset and all the features have data type "object".

    This is a challenging for the follwing reasons:

        - Some of these features have _large cardinalities_. Meaning, these featues have several thousands of unique values. We will have to choose feature encoding methods wisely and carefully to avoid/circumvent the problem of "curse of dimensionality".

        - _Novel Feature encoding_ 
    Feature encoding (converting categorical features into numeric values), by construction, creates extra features in our dataset. These extra features are dependent on and are proportional to the number of categories or unique values within the feature. This is not a problem when the number of categories in a feature is small. However, when the number of caterories (or unique values) gets larger (aka for features with large cardinality), the extra features created via feature encoding will become too much for the model to handle. This will greatly slow down computation time. This is situation is often referred to as "curse of dimensionality".

        - _Categorical Correlations_ 
    Computing correlations between categorical features cannot be done by the usual method of chi2 calculation. The mathematical/statistical method implemented here will be Cramer's V rule. We will be using an a new library called Dython and the function associations within dython to calculate the correlation matrix between catergorical variables. More details in the section under correlations.

        - This is a large dataset with about 118,000 data points (rows)!

    - _Tasks Performed:_
    
        - EDA
        Here we perform some exploratory data analysis to
        - understand the characteristics of the various features (columns) in our dataset
        - determine which columns we need to keep and which ones we can drop  
        - We will look like **null values, unique values and correlations** between the various features to identify the columns that need to be dropped.

  
        - _Feature engineering_.
          - Re-categorize some of the features to reduce the large number of unique values to a number that is more intuitive, useful and easier to handle.
          - We recast the date feature to better extract useful information.
     
        - _Feature encoding_.
      Different feature encoding methods were implemented here to properly tackle the extremely large cardinality in some of the features.

        - _ML Classification Models_.
          We applied and compares 4 differernt classification models to this datset, namely,
          - Logistic regression
          - KNN
          - Decision Trees (w and w/o bagging)
          - Random Forest
          We performed __hyperparameter tuning__ for some of the models using the _GridSearchCV_ approach. 

        _ _Error Analysis_.


__3. EDA_Fitbit_Fitness_Tracker__ 

    - _Data Source:_
    This data is obtained from Kaggle dataset (https://www.kaggle.com/datasets/arashnic/fitbit)

    - _Objective:_  
    This project was undertaken as part of the _first course of the IBM certification_, namely __Exploratory Data Analysis__. The objective of this project was to perform a compete and thorough __exploratory data analysis (EDA) on a dataset__.  

    - _Description:_ 
    This dataset was chosen from Kaggle and the analysis was done entirely by myself. EDA was performed on our fitbit fitness tracker dataset using the following methods:

    - _Understanding the nature of the data_
    This is done by studyign the shape and size of the dataset, scan for the null or missing values, understand the data types of the features etc.
  
    - _Hypothesis testing_
    Hypothesis testing is a very useful tool by which we can either verify or correct our own intuition about a dataset. this will help us make better decisions about how best to extract meaningful insights from our dataset.

    - _Data visualizations_
    Here we looked at the _correlations between the various features_ of the dataset, _checked for skewed data_ and corrected the skew where applied using _log transorfmation_ and finally looked for any _outliers_

    - _Feature engineering_
    Here we recast some of the features to make it better suited for future regression or classification analysis.



__4. LEGO_clustering__

    - _Objective:_ Image Segmentation
     We implemented 3 different clustering algorithms on a image of lego blocks of different colors to compare and identify which alorigm performs best. 
    

    - _Description:_ 
    This dataset is an RGB image of size 2178 x 3278 pixels that was taken from my personal phone camera. 

    - _Tasks Performed:_
        - Pre-processing : Imagine flattening.
        - _ML Clusering Model Comparison:_
            Three different unsupervised clustering algorithms were implemened and compared:
            - K-Means Clustering
            - Gaussian Mixture
            - Mean Shift 


