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

st.write("This App calculates how given parameters affect loan repayment based on provided csv or xlsx file and it also predicts probability of a client paying back loan")

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
        self.coef_ = np.zeros(n)  # Random initialization for coefficients
        self.intercept_ = 0.0  # Intercept is the first coefficient

    def fit(self, X, y, num_iterations=1000):
        
        
        

        
        for iteration in range(num_iterations):
            predictions = self._sigmoid(X.dot(self.coef_)+self.intercept_)
            
            
            errors = predictions - y
            gradient = X.T.dot(errors) / X.shape[0]
            

            self.coef_ = self.coef_ - self.learning_rate * gradient
            self.intercept_ = self.intercept_-self.learning_rate * np.sum(predictions - y) / X.shape[0]


    def _sigmoid(self, z):

        return 1 / (1 + np.exp(-z))



    def predict(self, X):       
        
        return self._sigmoid(X.dot(self.coef_)+self.intercept_)

x=LogReg(0.1, n=train.shape[1]-1)
x.fit(train.iloc[: , np.arange(train.shape[1]-1)],train.iloc[:,train.shape[1]-1])
st.title("Variable Coefficients")
st.write(x.coef_[0:2])

st.write("Data plot")
fig=plt.figure(figsize=(12,8))
sns.scatterplot(train,x="CCAvg", y="Income", hue="Personal.Loan")
xl=np.linspace(-1.5,3,1000)
plt.plot(xl, 0.04/2.15-0.42/2.15*(xl))
st.pyplot(fig)

st.title("Choose Credit Score")

# Slider
cr_score = st.slider("Select a value", min_value=round(min(train2.iloc[:,0]),1), max_value=round(max(train2.iloc[:,0]),1), value=float(max(train2.iloc[:,0])/2), step=0.1)

st.title("Choose Income")

# Slider
ind_inc = st.slider("Select a value", min_value=round(min(train2.iloc[:,1]),1), max_value=round(max(train2.iloc[:,1]),1), value=round(max(train2.iloc[:,1])/2), step=1)


my_prob=ss.transform(pd.DataFrame(data=[[cr_score, ind_inc ]], columns=['CCAvg', 'Income']))
prob=x.predict(my_prob)

st.title("Probability of repayment")
st.write(prob)  


