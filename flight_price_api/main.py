from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
# import sklearn
# print(sklearn.__version__)

preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('airline_price_model.pkl')

class Flight_detail(BaseModel):
    airline: str
    source_city: str
    destination_city: str
    departure_time: str
    arrival_time: str
    stops: str
    class_ : str
    days_left: int
    duration: Optional[float] = None

app = FastAPI()

@app.get("/")
def index():
    return {"Message": "Hello!"}

@app.get("/by-name/{name}")
def index(name):
    return {"Message": f"Hello {name}!"}

@app.post("/predict")
def predict_price(data: Flight_detail):
    try:
        # Convert the input data into a DataFrame
        input_data = pd.DataFrame([{
            'airline': data.airline,
            'source_city': data.source_city,
            'destination_city': data.destination_city,
            'departure_time': data.departure_time,
            'stops': data.stops,
            'class': data.class_,
            'days_left': data.days_left,
            'arrival_time' : data.arrival_time,
            'duration':data.duration
        }])
        
        processed_data = preprocessor.transform(input_data)
        # Make a prediction using the loaded model
        prediction = model.predict(processed_data )
        
        rounded_value =  round(prediction[0],2)
        # Return the prediction result
        return {"predicted_price": rounded_value}
    
    except Exception as e:
        # Log the error and return a 500 response
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")