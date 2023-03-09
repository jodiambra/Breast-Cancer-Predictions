import joblib 
def predict(data):
    model = joblib.load('Model/final_model.sav')
    return model.predict(data)

def predict_proba(data):
    model = joblib.load('Model/final_model.sav')
    return model.predict_proba(data)
