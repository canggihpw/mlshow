import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split

# import ML lib
from lib.svm import svm
from lib.logreg import logreg

from lib import _mldoc


@st.cache
def _get_dataset(ds):
    if ds == "Iris":
        iris = datasets.load_iris()
        X = iris.data[:, [0, 1]]
        y = iris.target
        label = iris.target_names
    elif ds == "Breast Cancer":
        cancer = datasets.load_breast_cancer()
        X = cancer.data[:,[0,1]]
        y = cancer.target
        label = cancer.target_names
    return X,y,label

# Demo or Documentation
dem = st.sidebar.radio("",("Demo","Learn"))
if dem == "Demo":
    # Select Dataset
    ds = st.sidebar.selectbox("Dataset",("Iris","Breast Cancer"),1)
    X,y,label = _get_dataset(ds)

    # Select test split
    testsize = st.sidebar.number_input("Test Size",0.2)
    rs = st.sidebar.number_input("Random State",1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=rs)


    # Select ML
    ml = st.sidebar.selectbox("ML Methods",
        ("SVM", "Logistic Regression")
    )
    if ml == "SVM":
        svm(X_train, X_test, y_train, y_test,label)
    elif ml == 'Logistic Regression':
        logreg(X_train, X_test, y_train, y_test,label)
else:
    _mldoc.mldoc()


# Plot
st.pyplot()



