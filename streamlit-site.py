import streamlit as st
from xgb import xgboost_eğitim
import pandas as pd
from gbm import  gbm_eğitim
import dummy

st.title("Effects of hyperparameters on the model")

df=pd.read_csv("kc_house_data.csv")
df=df.drop(columns={"id","date","zipcode","yr_renovated"},axis=1)


with st.sidebar:
    selected_option = st.selectbox("Select an option", ["LightGBM", "XGBoost"])
    st.header("Model Settings")

    max_depth = st.slider("Max Depth", min_value=1, max_value=10, value=3, help="The maximum depth of each tree in the XGBoost or LightGBM model. Increasing the max depth can make the model more complex and potentially overfit the data.")
    learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=1.0, value=0.01, help="The learning rate controls the step size at each boosting iteration. A lower learning rate makes the model learn more slowly but can lead to better generalization.")
    n_estimators = st.slider("Number of Estimators", min_value=1, max_value=1000, value=10, help="The number of boosting iterations or trees in the XGBoost or LightGBM model. Increasing the number of estimators can improve the model's performance, but also increase the training time.")
    böl = st.slider("Split Size", min_value=0.1, max_value=1.0, value=0.3, help="The split size determines the proportion of data used for training and testing. A higher split size means more data is used for training, while a lower split size means more data is used for testing.")
    st.write("[Github](https://github.com/onurincesu)")
    st.write("[Linkedin](https://www.linkedin.com/in/ali-onur-incesu-04bb59218/)")
def model_seç():
    if selected_option == "XGBoost":
        results = xgboost_eğitim(df,learning_rate,
                                    max_depth,
                                    n_estimators,böl)
        st.line_chart(results[0])
        st.write(results[1],"% Error rate")

    if selected_option=="LightGBM":
        results = gbm_eğitim(df,learning_rate,
                                    max_depth,
                                    n_estimators,böl)
        st.line_chart(results[0])
        st.write(results[1],"% Error rate")

model_seç()
