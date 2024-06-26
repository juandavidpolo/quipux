import joblib
from sklearn.datasets import load_iris

def prediction(features_data):
    response = response = {"success":True, "message":""}
    clf = joblib.load('./models/predict.joblib')
    data = load_iris()
    try:
        input_data = [features_data]
        prediction_result = clf.predict(input_data)
        result = data.target_names[prediction_result[0]]
        response["message"] = "Iris found"
        response["result"] = result
        return response
    except Exception as e:
        print("Error: ", str(e))
        response["success"] = False
        response["message"] = str(e)
        return response