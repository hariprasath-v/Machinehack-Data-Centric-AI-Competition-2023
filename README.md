# Machinehack-Data-Centric-AI-Competition-2023

## Public Leaderboard
* Rank : 15
* Score : 0.96096
## Private Leaderboard
* Rank : 16
* Score : 0.96096


### Competition hosted on <a href="https://machinehack.com/hackathons/data_centric_ai_competition_2023_image_data/overview">Machinehack</a>

### Problem
Build machine learning model to predict the character of each image..

### Evaluation
#### Evaluation metric for this competition is accuracy.

### Dataset

You can download the dataset <a href="https://machinehack.com/hackathons/data_centric_ai_competition_2023_image_data/data">here</a>    

### Solution

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


### Incorrect Label Image sample

#### Numbers
![Alt text](https://github.com/hariprasath-v/Machinehack-Data-Centric-AI-Competition-2023/blob/main/EDA%20%26%20Model%20Interpretation%20Visualization/Incorrect%20label%20number%20class.png)

#### Letters
![Alt text](https://github.com/hariprasath-v/Machinehack-Data-Centric-AI-Competition-2023/blob/main/EDA%20%26%20Model%20Interpretation%20Visualization/Incorrect%20label%20letter%20class.png)

### After removing 2834 miss labeled images from the train dataset, the cleaned data was trained with KNN classifier model with 1 nearest neighbor.
### KNN model gives good accuracy(0.8995) on validation data.

### ROC-AUC Score
![Alt text](https://github.com/hariprasath-v/Machinehack-Data-Centric-AI-Competition-2023/blob/main/EDA%20%26%20Model%20Interpretation%20Visualization/KNN%20ROC%20AUC%20score%20plot.png)

### True Positive Rate
![Alt text](https://github.com/hariprasath-v/Machinehack-Data-Centric-AI-Competition-2023/blob/main/EDA%20%26%20Model%20Interpretation%20Visualization/KNN%20True%20positive%20rate%20plot.png)

### <a href="https://github.com/hariprasath-v/Machinehack-Data-Centric-AI-Competition-2023/blob/main/Approach%20%26%20Solution%20-%20Data%20Centric%20AI%20Competition%202023%20%5BImage%20Data%5D.pdf">For more information about the model.</a>    

### Test Prediction

### Numbers
![Alt text](https://github.com/hariprasath-v/Machinehack-Data-Centric-AI-Competition-2023/blob/main/EDA%20%26%20Model%20Interpretation%20Visualization/Test%20prediction%20-%20Number%20class.png)

### Letters
![Alt text](https://github.com/hariprasath-v/Machinehack-Data-Centric-AI-Competition-2023/blob/main/EDA%20%26%20Model%20Interpretation%20Visualization/Test%20prediction%20-%20Letter%20class.png)


### File Information

data-centric-ai-competition-2023-image-data-eda.ipynb[![Open in Kaggle](https://img.shields.io/static/v1?label=&message=Open%20in%20Kaggle&labelColor=grey&color=blue&logo=kaggle)](https://www.kaggle.com/hari141v/data-centric-ai-competition-2023-image-data-eda)
 
data-centric-ai-competition-2023-image-data-model.ipynb[![Open in Kaggle](https://img.shields.io/static/v1?label=&message=Open%20in%20Kaggle&labelColor=grey&color=blue&logo=kaggle)](https://www.kaggle.com/code/hari141v/data-centric-ai-competition-2023-image-data-model)
 
 
   
        

