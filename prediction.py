import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "heartmodel.pkl",
    BASE_DIR / "Heart_model1.pkl",
]


def _load_model():
    for model_path in MODEL_CANDIDATES:
        if model_path.exists():
            return joblib.load(model_path)
    raise FileNotFoundError(
        "Could not find a trained model file. Expected one of: "
        + ", ".join(str(p.name) for p in MODEL_CANDIDATES)
    )


clfr = _load_model()

# Fit the scaler once using the same training split strategy used during prediction.
_df = pd.read_csv(BASE_DIR / "heart.csv")
_X = _df.drop("target", axis=1)
_y = _df["target"]
_X_train, _X_test, _y_train, _y_test = train_test_split(
    _X, _y, test_size=0.20, random_state=0
)
_scaler = StandardScaler()
_scaler.fit(_X_train)


def preprocess(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
 
    if sex.lower()=="male":
        sex=1 
    else: sex=0 
    if cp=="Typical angina":
        cp=0
    elif cp=="Atypical angina":
        cp=1
    elif cp=="Non-anginal pain":
        cp=2
    elif cp=="Asymptomatic":
        cp=3
    if exang=="Yes":
        exang=1
    elif exang=="No":
        exang=0
    if fbs=="Yes":
        fbs=1
    elif fbs=="No":
        fbs=0
    if slope=="Upsloping: better heart rate with excercise(uncommon)":
        slope=0
    elif slope=="Flatsloping: minimal change(typical healthy heart)":
          slope=1
    elif slope=="Downsloping: signs of unhealthy heart":
        slope=2  
    if thal=="fixed defect: used to be defect but ok now":
        thal=2
    elif thal=="reversable defect: no proper blood movement when excercising":
        thal=3
    elif thal=="normal":
        thal=1
    if restecg=="Nothing to note":
        restecg=0
    elif restecg=="ST-T Wave abnormality":
        restecg=1
    elif restecg=="Possible or definite left ventricular hypertrophy":
        restecg=2
    user_input=[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]
    user_input=np.array(user_input, dtype=float).reshape(1,-1)
    user_input=_scaler.transform(user_input)
    prediction = clfr.predict(user_input)
    result = int(prediction[0])

    return result

# if __name__ == '__main__':
#     t=preprocess(39,"male","Non-anginal pain",130,"ST-T Wave abnormality",250,"Yes",187,"Yes",2,"Downsloping: signs of unhealthy heart",2,"normal")
#     print(int(t))