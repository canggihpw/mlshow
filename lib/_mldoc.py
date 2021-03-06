import streamlit as st

def mldoc():
    st.title("Machine Learning Model Evaluation")
    st.header("Classification Problem")

    st.subheader("Confusion Matrix")
    st.markdown('''
    |-|Actual 1|Actual 0|
    |-----|-----|-----|
    |Predict 1|TP|FP|
    |Predict 0|FN|TN|
    ''')
    
    st.subheader("Accuracy")
    st.latex(r'\frac{TP + TN}{TP + FP + TN + FN}')

    # untuk outlier, TN tidak penting, pakenya:
    # precision, recall, f1-score, PR AUC
    # untuk klasifikasi 2 kelas biasa, TN penting:
    # TPR, FPR, ROC AUC

    st.subheader("Precision")
    st.latex(r'\frac{TP}{TP + FP}')
    st.markdown('Menunjukkan')

    st.subheader("Recall/Sensitivity/True Positive Rate")
    st.latex(r'\frac{TP}{TP + FN}')

    st.subheader("Specifity")
    st.latex(r'\frac{TN}{TN + FP}')

    st.subheader("False Positive Rate")
    st.latex(r'1 - specifity = \frac{FP}{TN + FP}')

    st.subheader("F1-score")
    st.latex(r'\frac{2}{\frac{1}{precision} + \frac{1}{recall}}')

    st.subheader("Precision-Recall Curve--AUC")

    st.subheader("Receiver Operating Characteristic (ROC) Curve--AUC")


