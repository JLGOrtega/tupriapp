from sqlalchemy import create_engine
from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import datetime
import json
import io
import base64
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import google.generativeai as genai
from utils import get_prompt, get_ts, generar_texto

load_dotenv()


app = Flask(__name__)

churro = os.environ["CHURRO"]
engine = create_engine(churro)


@app.route('/', methods=["GET"])
def formulario():
    return render_template('formulario.html')

@app.route("/predict", methods=["POST"])
def predict():
    #RECOGEMOS LOS INPUTS
    pclass = int(request.form.get("pclass"))
    sex = int(request.form.get("sex"))
    age = int(request.form.get("age"))

    inputs = [pclass, sex, age]

    # CARGAMOS EL MODDELO
    with open("titanic_model.pkl", "rb") as f:
        modelito = pickle.load(f)
    
    # HACEMOS PREDICCIONES Y MONTAMOS EL TIMESTAMP
    outputs = modelito.predict([inputs])[0]
    timestamp = get_ts()

    # PARRIBA
    logs_to_parriba = pd.DataFrame({"inputs": [inputs], 
                                    "predictions": [outputs], 
                                    "timestamps": [timestamp]})
    logs_to_parriba.to_sql("predictions", con=engine, index=False, if_exists="append")


    logs_leidos = pd.read_sql("""SELECT * FROM predictions""", con=engine)
    
    fig = plt.figure()
    logs_leidos.predictions.value_counts().plot(kind="bar")
    plt.title(f"PREDICTIONS UP TO : {logs_leidos.timestamps.max()}")

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close(fig)

    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')


    # GENERAR TEXTO IA
    
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])  # Reemplaza con tu clave real
    # Define el modelo a usar
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    prompt = get_prompt(inputs, outputs)
    generacion = generar_texto(model, prompt, temperature=0.7, top_p=1.0, top_k=40, max_output_tokens=512)

    return render_template('resultado.html', 
                           prediccion=outputs, 
                           grafica=img_base64, 
                           texto_ia=generacion)

@app.route("/results", methods=["GET"])
def results():
    logs_leidos = pd.read_sql("""SELECT * FROM predictions""", con=engine)
    return json.loads(logs_leidos.to_json(orient="records"))


if __name__ == "__main__":
    app.run(debug=True)