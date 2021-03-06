import streamlit as st

from lib._plot import plot_decision_regions
from sklearn.svm import SVC

def svm(X_train, X_test, y_train, y_test,label):
    pil = st.sidebar.radio("",("Result","Documentation"))
    if pil == "Result":
        # Controller
        gammaval = st.sidebar.slider('gamma', 0.1, 1.0, 1.0)
        cval = st.sidebar.selectbox("C",(1, 10, 100, 1000),0)
        kernel = st.sidebar.selectbox("Kernel",('linear', 'poly','rbf','sigmoid'),0)
        degree = st.sidebar.number_input("Degree (d)",1)
        coef0 = st.sidebar.number_input("Coef0 (r)",0.0)

        clf = SVC(gamma=gammaval, C =cval, kernel=kernel, degree=degree, coef0=coef0,probability=True)
        clf.fit(X_train, y_train)

        st.title("SVM Classification Result")
        plot_decision_regions(X_test,y_test,clf,len(y_train),label)
    elif pil == "Documentation":
        st.title("SVM Documentation")
        documentation()

def documentation():
    st.header("A. Main theoretical expression")
    
    st.subheader("Basic Decision Rule")
    st.latex(r'\mathbf{w} \cdot \mathbf{u} + b \geq 0, \textnormal{then } \oplus')

    st.subheader("Gutter")
    st.latex(r'y_i(\mathbf{x}_i \cdot \mathbf{w} + b) - 1 = 0, \textnormal{for } \mathbf{x}_i \textnormal{ in gutter}')

    st.subheader("Boundary Width")
    st.latex(r'(\mathbf{x}_{\oplus} - \mathbf{x}_{\ominus}) \cdot \frac{\mathbf{w}} {\| \mathbf{w} \|} = \frac{2}{\| \mathbf{w} \|}')

    st.subheader("Optimal w")
    st.latex(r'\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i')

    st.subheader("Objective Function Optimal")
    st.latex(r'L = \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_j \alpha_i \alpha_j y_i y_j \mathbf{x}_i \cdot \mathbf{x}_j')

    st.subheader("Optimal Decision Rule")
    st.latex(r'\sum_i \alpha_i y_i \mathbf{x}_i \cdot \mathbf{v} + b \geq 0, \textnormal{then } \oplus')

    st.subheader("Kernel Function")
    st.latex(r'K(\mathbf{x}_i,\mathbf{x}_j) = \phi(\mathbf{x}_i) \cdot \phi(\mathbf{x}_j)')

    #------------------

    st.header("B. Implementation")
    st.subheader("Linear kernel")
    st.latex(r'K(\mathbf{x}_i,\mathbf{x}_j) = \langle \mathbf{x}_i,\mathbf{x}_j \rangle')

    st.subheader("Polynomial kernel")
    st.latex(r'K(\mathbf{x}_i,\mathbf{x}_j) = (\gamma \langle \mathbf{x}_i,\mathbf{x}_j \rangle + r)^d')

    st.subheader("RBF kernel")
    st.latex(r'K(\mathbf{x}_i,\mathbf{x}_j) = e^{-\gamma \| \mathbf{x}_i - \mathbf{x}_j \|^2}')

    st.subheader("Sigmoid kernel")
    st.latex(r'K(\mathbf{x}_i,\mathbf{x}_j) = \tanh(\gamma \langle \mathbf{x}_i,\mathbf{x}_j \rangle + r)')


    #------------------

    st.header("C. Parameter Description")
    st.markdown('''
    |parameters|description|
    |-----|-----|
    |gamma|defines how much influence a single training example has. The larger gamma is, the closer other examples must be to be affected.|
    |C|The parameter C, common to all SVM kernels, trades off misclassification of training examples against simplicity of the decision surface. A low C makes the decision surface smooth, while a high C aims at classifying all training examples correctly.|
    |kernel|linear,poly,rbf,sigmoid|
    |degree|degree of polynomial kernel|
    |coef0|independent coefficient in polynomial & sigmoid kernel|
    
    ''')