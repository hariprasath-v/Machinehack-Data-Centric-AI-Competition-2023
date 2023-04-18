# Machinehack-Data-Centric-AI-Competition-2023


### Competition hosted on <a href="https://machinehack.com/hackathons/data_centric_ai_competition_2023_image_data/overview">Machinehack</a>

### Problem
Build machine learning model to predict the character of each image..

### Evaluation
#### Evaluation metric for this competition is accuracy.

### Dataset

You can download the dataset <a href="https://machinehack.com/hackathons/data_centric_ai_competition_2023_image_data/data">here</a>    

### Solution:

### Exploratory Data Analysis

#### The basic exploratory data analysis of the data,
* Basic image meta data analysis
* Image similarity analysis

#### The above analysis had done by using,
* cv2 
* Image
* numpy
* seaborn
* matplotlib
* pandas

### Model
#### The train dataset contains miss labeled images, the incorrectly labeled images are identified by using ,

  * Simple pytorch convolutional neural network
  * Skorch - Scikit-Learn compatible neural network library
  * Cleanlab - No code library to fix the errors in dataset

The simple CNN model run 4 iterations(50 epochs/iteration) and cross-validated on
10-kfold split train dataset.In each iteration, the actual label and the predicted
probability for the labels are compared by using the find_label_issue function from
cleanlab tool.Then, based on the self-confidence level the miss labeled images are
collected.


### Trained KNN classifier with 1 nearest neighbor.

![Alt text](-)

![Alt text](-)


### File information

eda.ipynb[![Open in Kaggle](https://img.shields.io/static/v1?label=&message=Open%20in%20Kaggle&labelColor=grey&color=blue&logo=kaggle)](-)
 
model.ipynb[![Open in Kaggle](https://img.shields.io/static/v1?label=&message=Open%20in%20Kaggle&labelColor=grey&color=blue&logo=kaggle)](-)
 
 
   
        

