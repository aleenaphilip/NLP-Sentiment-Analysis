# NLP-Sentiment-Analysis

Objective: 

The goal of this project is to develop a model that can classify sentiments in healthcare reviews. This involves analyzing text data from healthcare reviews and determining whether the sentiment expressed in each review is positive, negative, or neutral.

Workflow :

 1. Data Preprocessing
 2. Sentiment Analaysis Model
 3. Model Evaluation
 4. Insight and visualization


Libraries Used:

import pandas as pd

import numpy as np

import nltk

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string  

from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.neighbors import KNeighborsClassifier 

import matplotlib.pyplot as plt 

import seaborn as sns 

import plotly.express as px

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from imblearn.over_sampling import RandomOverSampler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
