# Model eğitimi

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib as plt
import pandas as pd

def gbm_eğitim(df,l_rate,m_depht,n_est,böl):
    y = df["price"]
    X = df.drop("price", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=böl, random_state=1)
    model = lgb.LGBMRegressor(learning_rate=l_rate,
                              max_depth=m_depht,
                              n_estimators=n_est,
                              verbosity=-1
                             ).fit(X_train, y_train)
    
    preds=model.predict(X_test)
    results=chartgbm(y_test,preds)
    return results

def chartgbm(y_test,preds):
    veri=pd.DataFrame({"Real":y_test,"Predicted":preds}).sample(n=100,random_state=42)
    hata_orani = np.abs(veri["Real"] - veri["Predicted"]) / veri["Real"] * 100
    hata_orani=round(sum(hata_orani) / len(hata_orani), 4)
    return veri,hata_orani

    