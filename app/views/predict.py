from fastapi import APIRouter, HTTPException, Body
from app.controllers.predict import prediction
from pydantic import BaseModel, ValidationError as PydanticValidationError
import numbers

router = APIRouter()

class Features(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@router.post("/predict")
def read_user(features: Features):
    features = {
        "sepal_length": features.sepal_length,
        "sepal_width": features.sepal_width,
        "petal_length": features.petal_length,
        "petal_width": features.petal_width
    }
    print(features)
    validate = validation(features)
    if not validate["success"]:
        raise HTTPException(status_code=400, detail=validate["message"])
    try:
        features_data = [features["sepal_length"], features["sepal_width"], features["petal_length"], features["petal_width"]]
        response = prediction(features_data)
        return response
    except PydanticValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def validation(features):
    required_fields = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for field in required_fields:
        parse_field = field.replace("_", " ")
        if field not in features:
            return {"success": False, "message": f"{parse_field.capitalize()} is required"}
        elif not isinstance(features[field], numbers.Number) or features[field] < 0:
            return {"success": False, "message": f"{parse_field.capitalize()} must be a number and greater than zero"}
    return {"success": True}