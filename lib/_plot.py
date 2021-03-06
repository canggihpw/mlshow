import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report

def plot_decision_regions(X,y,clf,numtrain,labels):
    crep = classification_report(y, clf.predict(X), target_names=labels,output_dict=True)
    #------------------------------
    # Plotting decision regions
    st.header("Decision Boundary")
    st.write("Total Accuracy = " + str(round(crep['accuracy'],4)))
    st.write("Training | Testing = " + str(numtrain) + " | " + str(len(y)))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),np.arange(y_min, y_max, 0.1))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y,s=20, edgecolor='k')
    st.pyplot()

    #------------------------------
    # Plotting confusion matrix
    st.header("Confusion Matrix")
    plot_confusion_matrix(clf, X, y)
    st.pyplot()

    #------------------------------
    # Plotting metrics
    st.header("Classification Metrics")
    data = []
    for label in labels:
        lbl = crep[label]
        lbl['class'] = label
        data.append(lbl)
    dfc = pd.DataFrame(data)
    dfc = dfc.set_index('class')
    st.dataframe(dfc)