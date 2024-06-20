# Machine Learning Projects

I worked on these projects as part of the __IBM Machine Learning Professional Certification__. 

__1. Deep Learning Project : GAN for generating handwritten images using MNIST Data__

    - Data Source:
    MNIST dataset 

    - Objective:
    To train a DCGAN (Deep Convolutional Generative Adversarian Network) to generate handwritten images.

    - Description: 
    
        1. We use a GAN (Generative Adversarial neural Network) comprising of two deep convolutional networks
        
            A. GENERATOR for upsampling the data from noise to a AI generated 28x28 image)
            
                • Activation function (Network Layers) : ReLu 
                • Activation function (Generator Output) : tanh (to keep the values of the generated image between -1 and 1
                • Number of layers : 11 layers
                
            B. DISCRIMINATOR (for downsampling the generated data to a single (1x1) output
            
                • Activation function (network layers): leaky ReLU
                • Activation function (Discriminator Output) : Sigmoid activation is applied to the discriminator via the sigmoid Binary Cross Entropy (BCE) Loss function
                • Number of layers : 9 layers
                
        2. We use a kernel size of 5x5 for the generator and discriminator
        
        3. Loss function : BCE (Binary Cross Entropy)

    - Tasks Performed:
    
        1. Built a GAN model comprising of two separate and independant concolutional neural networks, namely
            
            - the Generator : to generate images 
            - the Discriminator : to distinguish a generated image from a real image 
            
        2. Defined Loss Functions for the Generator and the Discriminator

        3. Create a traning function to do the following:
            a. start by sampling a random normal noise vector
            b. generating an image (Xgen) by passing the noise to the generator
            c. pass the generated image (Xgen) along with the real image (Xreal) into the discriminator
            d. compute generator loss, L_g, and the total discriminator loss, L_d by comparing the output of the discriminator to the real and fake images. 
            e. compute gradients for the generator and the discriminator using the loss computed above
            f. update the generator and discriminator optimizers defined above (gen_opt and dis_opt) using the gradients 

        4. Compute the Error Rate/Accuracy of the entire model

        5. Hyperparameter Tuning
            - No. of Epochs Vs Error Rate
            - Batch Size Vs Error Rate
            - Noise Vs Error Rate

__2. Supervised Learning- Classification : FDA_Drug_Classification__

    - Data Source:
    https://www.kaggle.com/datasets/maheshdadhich/us-healthcare-data?datasetId=7684&sortBy=dateRun&tab=profile 

    - Objective:
    The objective of this project is to predict the catergory of any given drug in terms of one of the following

        - OTC (over the counter)
        - Prescription
        - Allergenic
          or

        - Other
  
    This is a __classification problem__ and we shall implement several different types of classification models to make the above prediction.


    - Description: 

    This dataset is interesting for couple of its intriguing characteristics, namely, __ALL the features of this dataset are categorical!__. That is, there are no numerical values in this dataset and all the features have data type "object".

    This is a challenging for the follwing reasons:

        - Some of these features have _large cardinalities_. Meaning, these featues have several thousands of unique values. We will have to choose feature encoding methods wisely and carefully to avoid/circumvent the problem of "curse of dimensionality".

        - Novel Feature encoding 
    Feature encoding (converting categorical features into numeric values), by construction, creates extra features in our dataset. These extra features are dependent on and are proportional to the number of categories or unique values within the feature. This is not a problem when the number of categories in a feature is small. However, when the number of caterories (or unique values) gets larger (aka for features with large cardinality), the extra features created via feature encoding will become too much for the model to handle. This will greatly slow down computation time. This is situation is often referred to as "curse of dimensionality".

        - Categorical Correlations 
    Computing correlations between categorical features cannot be done by the usual method of chi2 calculation. The mathematical/statistical method implemented here will be Cramer's V rule. We will be using an a new library called Dython and the function associations within dython to calculate the correlation matrix between catergorical variables. More details in the section under correlations.

        - This is a large dataset with about 118,000 data points (rows)!

    - Tasks Performed:
    
        - EDA
        Here we perform some exploratory data analysis to
        - understand the characteristics of the various features (columns) in our dataset
        - determine which columns we need to keep and which ones we can drop  
        - We will look like **null values, unique values and correlations** between the various features to identify the columns that need to be dropped.

  
        - Feature engineering.
          - Re-categorize some of the features to reduce the large number of unique values to a number that is more intuitive, useful and easier to handle.
          - We recast the date feature to better extract useful information.
     
        - Feature encoding.
      Different feature encoding methods were implemented here to properly tackle the extremely large cardinality in some of the features.

        - ML Classification Models.
          We applied and compares 4 differernt classification models to this datset, namely,
          - Logistic regression
          - KNN
          - Decision Trees (w and w/o bagging)
          - Random Forest
          We performed __hyperparameter tuning__ for some of the models using the _GridSearchCV_ approach. 

        _ Error Analysis.


__3. EDA_Fitbit_Fitness_Tracker__ 

    - Data Source:
    This data is obtained from Kaggle dataset (https://www.kaggle.com/datasets/arashnic/fitbit)

    - Objective:  
    This project was undertaken as part of the _first course of the IBM certification_, namely __Exploratory Data Analysis__. The objective of this project was to perform a compete and thorough __exploratory data analysis (EDA) on a dataset__.  

    - Description: 
    This dataset was chosen from Kaggle and the analysis was done entirely by myself. EDA was performed on our fitbit fitness tracker dataset using the following methods:

    - Understanding the nature of the data
    This is done by studyign the shape and size of the dataset, scan for the null or missing values, understand the data types of the features etc.
  
    - Hypothesis testing
    Hypothesis testing is a very useful tool by which we can either verify or correct our own intuition about a dataset. this will help us make better decisions about how best to extract meaningful insights from our dataset.

    - Data visualizations
    Here we looked at the _correlations between the various features_ of the dataset, _checked for skewed data_ and corrected the skew where applied using _log transorfmation_ and finally looked for any _outliers_

    - Feature engineering
    Here we recast some of the features to make it better suited for future regression or classification analysis.



__4. LEGO_clustering__

    - Objective: Image Segmentation
     We implemented 3 different clustering algorithms on a image of lego blocks of different colors to compare and identify which alorigm performs best. 
    

    - Description: 
    This dataset is an RGB image of size 2178 x 3278 pixels that was taken from my personal phone camera. 

    - Tasks Performed:
        - Pre-processing : Imagine flattening.
        - _ML Clusering Model Comparison:_
            Three different unsupervised clustering algorithms were implemened and compared:
            - K-Means Clustering
            - Gaussian Mixture
            - Mean Shift 


