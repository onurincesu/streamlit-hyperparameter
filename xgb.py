
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pandas as pd


def xgboost_eğitim(df,l_rate,m_depth,n_est,böl):
    
    y = df["price"]
    X = df.drop("price", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=böl, random_state=1)
    model = XGBRegressor(learning_rate=l_rate,
                        max_depth=m_depth,
                        n_estimators=n_est).fit(X_train,y_train)
    
    
    preds=model.predict(X_test)
    results=chartxgb(y_test,preds)
    return results
    
def chartxgb(y_test,preds):
    veri=pd.DataFrame({"Gerçek":y_test,"Tahmin":preds}).sample(n=100,random_state=42)
    hata_orani = np.abs(veri["Gerçek"] - veri["Tahmin"]) / veri["Gerçek"] * 100
    hata_orani=round(sum(hata_orani) / len(hata_orani), 4)
    return veri,hata_orani
