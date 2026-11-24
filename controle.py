import joblib, pandas as pd
m = joblib.load("models/model.pkl")
X = pd.DataFrame([{"montant": 1800, "anciennete": 30, "freq_24h": 4, "nuit": 1}])
print(round(m.predict_proba(X)[0, 1], 4))
