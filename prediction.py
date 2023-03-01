import joblib 
def predict(data):
    model = joblib.load('Model/final_model.sav')
    return model.predict(data)