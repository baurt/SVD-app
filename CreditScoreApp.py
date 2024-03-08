import streamlit as st
from skimage import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris # pip install scikit-learn
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler

st.write("This App predicts probability of a client paying back loan")

st.title("Upload your data to train the app")

# File Upload
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the file
    try:
        if uploaded_file.name.endswith('.csv'):
            train = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            train = pd.read_excel(uploaded_file, engine='openpyxl')
        
        # Display the data
        s#t.write("Uploaded File Contents:")
        #st.write(train)
        
    except Exception as e:
        st.error(f"Error: {e}")
else:
    train = pd.read_csv('credit_train.csv')
train2=train.copy()

#scaling data
ss=StandardScaler()
train.iloc[: , np.arange(train.shape[1]-1)]=ss.fit_transform(train.iloc[: , np.arange(train.shape[1]-1)])

#defining logreg class and its methods
class LogReg:
    def __init__(self, learning_rate=0.001, n=1):
        self.learning_rate = learning_rate
        self.n_features = n
        self.coef_ = np.random.uniform(size=(n + 1))  # Random initialization for coefficients
        self.intercept_ = self.coef_[0]  # Intercept is the first coefficient

    def fit(self, X, y, num_iterations=1000):
        # Add a column of ones to X for the intercept term
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        

        
        for iteration in range(num_iterations):
            predictions = 1/(1+np.exp(X_with_intercept.dot(self.coef_)))
            
            
            errors = predictions - y
            gradient = X_with_intercept.T.dot(errors) / X.shape[0]
            

            self.coef_ = self.coef_ - self.learning_rate * gradient
            self.intercept_ = self.coef_[0]



    def predict(self, X):
        X_with_intercept = np.c_[np.ones(X.shape[0]), X]
        
        return 1-1/(1+np.exp(X_with_intercept.dot(self.coef_)))

x=LogReg(0.1, n=train.shape[1]-1)
x.fit(train.iloc[: , np.arange(train.shape[1]-1)],train.iloc[:,train.shape[1]-1])

st.title("Choose Credit Score")

# Slider
cr_score = st.slider("Select a value", min_value=round(min(train2.iloc[:,0]),1), max_value=round(max(train2.iloc[:,0]),1), value=float(max(train2.iloc[:,0])/2), step=0.1)

st.title("Choose Income")

# Slider
ind_inc = st.slider("Select a value", min_value=round(min(train2.iloc[:,1]),1), max_value=round(max(train2.iloc[:,1]),1), value=round(max(train2.iloc[:,1])/2), step=1)

x.predict(train[["CCAvg","Income"]])

my_prob=ss.transform(pd.DataFrame(data=[[cr_score, ind_inc ]], columns=['CCAvg', 'Income']))
prob=x.predict(my_prob)
st.write(prob)  


