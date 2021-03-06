import streamlit as st

from lib._plot import plot_decision_regions
from sklearn.linear_model import LogisticRegression

def logreg(X_train, X_test, y_train, y_test,label):
    # Controller
    penalty = st.sidebar.selectbox("Penalty",('l1', 'l2', 'elasticnet'),1)
    solver = st.sidebar.selectbox("Solver",('newton-cg', 'lbfgs', 'liblinear','sag','saga'),1)

    clf = LogisticRegression(penalty=penalty,solver=solver)
    clf.fit(X_train, y_train)
    plot_decision_regions(X_test,y_test,clf,len(y_train),label)