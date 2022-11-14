# import Libaraires

import streamlit as st 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import datasets 
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# heading
st.write("""
# Machine Leaning Web-application in python with streamlit
""")
st.write("""
# Explore different ML Models and Datasets 
To check the Best one 
""")

# data set put in sidebar dataset
dataset_name = st.sidebar.selectbox(
    'Select Data Set',
    ('Iris','Breast Cancer','Wine')
)

# data set put in sidebar for classsifier
classifier_name = st.sidebar.selectbox(
    'Select Data Set',
    ('KNN','SVM','Random Forest')
)

#Function for Datasets

def get_dataset(dataset_name):
    data=None
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "wine":
        data =datasets.load_wine()
    else:
        data= datasets.load_breast_cancer()
    x = data.data
    y = data.target

    return x,y

# call function 
X,y =get_dataset(dataset_name)

st.write('Shape of dataset:',X.shape)
st.write('Numbe of classes:',len(np.unique(y)))

# next different classifier

def add_parameter_ui(classifier_name):
    params = dict() # create empty dictionary
    if classifier_name == 'SVM':
        C = st.sidebar.slider('C', 0.01,10.0)
        params['C'] = C # its  the degree of correct classfiifcation
    elif classifier_name == 'KNN':
        K= st.sidebar.slider('K', 1,15)
        params['K'] = K # its the number of nerest neighbours
    else:
        max_depth = st.sidebar.slider('max_depth',2,15)
        params['max_depth'] =max_depth # depth of every treee that grow in random forest
        n_estimators =st.sidebar.slider('n_estimators',1,100)
        params['n_estimators'] = n_estimators # number of tress
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(classifier_name,params):
    clf=None
    if classifier_name == 'SVM':
        clf =SVC(C=params['C'])
    elif classifier_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    else:
        clf = clf= RandomForestClassifier(n_estimators=params['n_estimators'],
         max_depth=params['max_depth'],random_state=1234)

    return clf

clf =get_classifier(classifier_name,params)

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=1234)

clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

# check accuracy
acc = accuracy_score(y_test, y_pred)
st.write(f'Classifier={classifier_name}')
st.write(f'Accuracy=',acc)

# Plot Dataset

pca = PCA(2)
X_projected = pca.fit_transform(X)

# data 0 ,1  dimission slice
x1= X_projected[:,0]
x2 = X_projected[:,1]

fig =plt.figure()
plt.scatter(x1,x2,
          c=y,
          cmap='viridis')  
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# show plot 
st.pyplot(fig)


