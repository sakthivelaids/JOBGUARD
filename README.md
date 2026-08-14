## Problem
Scammers advertise jobs the same way legitimate employers do — online (in ads, on job sites, and social media), in newspapers, and sometimes on TV and radio. They promise you a job, but what they want is your money and your personal information.

## Model Description 
Fake-Job-Predictor is a python based machine learning model, which uses algorithms like the Naive Bayes algorithm and Decision tree classifier algorithm, to predict whether a posted job is fake or real. The model is deployed over heroku and can be viewed [here.](https://fake-job-prediction.herokuapp.com/)

Code for all the detailed analysis and model building can be viewed [here](Fake_job_predictor.ipynb).

<p align="center">
  <img width="460" height="300" src="Images/image1.PNG">
</p>

### Technology Stacks and Libraries

* Numpy
* Pandas
* matplotlib
* Imbalanced-learn
* wordcloud
* Natural Language Toolkit
* Multinomial Naive Bayes (scikit-learn)
* Decision tree classifier (scikit-learn)
* flask

## Analysis of Dataset
The Model uses Employment Scam Aegean Dataset (EMSCAD) , which can be viewed [here.](https://www.kaggle.com/amruthjithrajvr/recruitment-scam)

### Visualizing dataset based on location 

<p align="center">
  <img width="460" height="300" src="Images/image2.PNG">
</p>

### Visualizing dataset based on Experience

<p align="center">
  <img width="460" height="300" src="Images/image3.PNG">
</p>

### Visualizing common words used in the dataset using Wordcloud

<p align="center">
  <img width="460" height="300" src="Images/image4.PNG">
</p>

## Model Evaluation 

### Naive Bayes Approach 
In the first approach, Multinomial Naive Bayes,which is one of the most popular supervised learning classifications was used for the analysis of the categorical text data. <br>
Classification Accuracy: 0.8980769230769231 <br>
Confusion Matrix:
<p align="center">
  <img width="460" height="300" src="Images/image5.PNG">
</p>

### Decision tree Classifier
In the Second approach, Decision tree classifier was used which classifies inputs by segmenting the input space into regions. <br>
Classification Accuracy: 0.8173076923076923 <br>
Confusion Matrix:
<p align="center">
  <img width="460" height="300" src="Images/image6.PNG">
</p>

## Streamlit App

I added a Streamlit web app to train and run the fake-job predictor locally.

- App file: [app.py](app.py)
- Helper: [model_utils.py](model_utils.py)
- Requirements: [requirements.txt](requirements.txt)

Quick start (create a virtualenv first):

```powershell
pip install -r requirements.txt
streamlit run app.py
```

Notes:
- Upload the dataset CSV (the same one used in the notebook) when prompted in the app.
- The app trains a simple pipeline (TF-IDF + MultinomialNB or DecisionTree) in-browser and can predict a single job posting text.

Docker (build and run)

Build the image from the project folder `Fake-Job-Prediction-Model-main` (make sure `Dockerfile` is present there):

```powershell
docker build -t fake-job-predictor .
docker run -p 8501:8501 fake-job-predictor
```

Then open http://localhost:8501 in your browser.

If the Docker build fails due to native build dependencies (for packages like `wordcloud`), try installing the system packages listed by the error (for Debian/Ubuntu: `build-essential`, `libjpeg-dev`, `zlib1g-dev`, `libfreetype6-dev`) or build on a machine with the needed wheels available.

