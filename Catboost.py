
import streamlit as st
# base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Важная настройка для корректной настройки pipeline!
import sklearn
sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import GridSearchCV, KFold

# for model learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

#models
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

# Metrics
from sklearn.metrics import accuracy_score


# tunning hyperparamters model



df = pd.read_csv('heart.csv')
X, y = df.drop('HeartDisease', axis=1), df['HeartDisease']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


####
###Imputer
my_imputer = ColumnTransformer(
    transformers = [
        
        ('num_imputer', SimpleImputer(strategy='mean'), ['Age']), # SimpleImputer Позволяет заполнить каким-либо простым показателем (средним, модой, медианой)
        
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough' 
)   

#####
###Scaler
ordinal_encoding_columns = ['Sex', "FastingBS", "ExerciseAngina"] # Столбец, который планируем кодировать порядково, с помощью OrdinalEncoder 
one_hot_encoding_columns = ['ChestPainType', "RestingECG", "ST_Slope"] # Столбец, который планируем кодировать с помощью OneHotEncoder 

standard_scaler_columns = ['Age', 'RestingBP',"Cholesterol","MaxHR", "Oldpeak"] # Числовые столбцы, которые необходимо пронормировать


scaler_and_encoder = ColumnTransformer(
    [
        ('ordinal_encoding', OrdinalEncoder(), ordinal_encoding_columns),
        ('one_hot_encoding_columns', OneHotEncoder(sparse_output=False), one_hot_encoding_columns),
        ('scaling_num_columns', StandardScaler(), standard_scaler_columns)
    ],
    verbose_feature_names_out = False,
    remainder = 'passthrough' 
) 

#####
####Preprocessor
preprocessor = Pipeline(
    [
        ('imputer', my_imputer),
        ('scaler_and_encoder', scaler_and_encoder)
    ]
)
from catboost import CatBoostClassifier
preprocessor.fit(X_train, y_train)
preprocessed_X_train = preprocessor.transform(X_train)
preprocessed_X_valid = preprocessor.transform(X_valid)
cb = CatBoostClassifier(eval_metric='Accuracy')

cb.fit(
    preprocessed_X_train,
    y_train,
    eval_set=(preprocessed_X_valid, y_valid)
)
train3=accuracy_score(y_train, cb.predict(preprocessed_X_train))
valid3=accuracy_score(y_valid, cb.predict(preprocessed_X_valid))
print('train accuracy:', train3) # Доля правильных ответов на выборке, которая была использована для обучения
print('valid accuracy:', valid3) # Доля правильных ответов на выборке, которую обученный алгоритм еще не видел

st.write("This App predicts if a person has heartdisease based on given parameters")



# Slider
Age = st.sidebar.slider("Select age", min_value=0, max_value=100, value=50, step=1)

Sex = st.sidebar.selectbox('Select gender', df["Sex"].drop_duplicates().to_list())
ChestPain = st.sidebar.selectbox('Select chest pain type', df["ChestPainType"].drop_duplicates().to_list())
RestingBp=st.sidebar.slider("Select resting blood pressure", min_value=min(df["RestingBP"]), max_value=max(df["RestingBP"]), value=100, step=1)
Cholesterol=st.sidebar.slider("Select cholesterol", min_value=min(df["Cholesterol"]), max_value=max(df["Cholesterol"]), value=150, step=1)
Fasting = st.sidebar.selectbox('Select FastingBS', df["FastingBS"].drop_duplicates().to_list())
RestingECG = st.sidebar.selectbox('Select RestingECG', df["RestingECG"].drop_duplicates().to_list())
MaxHR=st.sidebar.slider("Select Maximum Heart Beat Rate", min_value=min(df["MaxHR"]), max_value=max(df["MaxHR"]), value=150, step=1)
Angina = st.sidebar.selectbox('Exercise Angina?', df["ExerciseAngina"].drop_duplicates().to_list())
Oldpeak=st.sidebar.slider("Select Oldpeak", min_value=round(min(df["Oldpeak"]),1), max_value=round(max(df["Oldpeak"]),1), value=1.0, step=0.1)
Stslope = st.sidebar.selectbox('Select the slope of peak exercise', df["ST_Slope"].drop_duplicates().to_list())


df2=pd.DataFrame({"Age":[Age], "Sex":[Sex], "ChestPainType":[ChestPain], "RestingBP":[RestingBp],"Cholesterol":[Cholesterol],"FastingBS":[Fasting],"RestingECG":[RestingECG],
                  "MaxHR":[MaxHR],"ExerciseAngina":[Angina],"Oldpeak":[Oldpeak],"ST_Slope":[Stslope]})
my_prob=preprocessor.transform(df2)

prob=cb.predict(my_prob)
st.title("Recommendations:")
if prob==0:
    st.write("The app predicts that the person is healthy")
else:
    st.write("The person might need to see a doctor")
