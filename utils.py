import datetime
import google.generativeai as genai

def get_ts():
    timestamp = datetime.datetime.now().isoformat()
    return timestamp[0:19]


def generar_texto(model, prompt, temperature=0.7, top_p=1.0, top_k=40, max_output_tokens=512):
    """
    Genera texto usando el modelo Gemini 2.0 Flash.

    Args:
        prompt: El texto de entrada (prompt) para el modelo.
        temperature: Controla la aleatoriedad de la generación.
                     Valores más altos hacen la salida más creativa.
        top_p: Controla la probabilidad acumulativa de las tokens. 
               Valores más altos seleccionan más tokens.
        top_k:  Controla la cantidad de tokens más probables a considerar.
        max_output_tokens: El número máximo de tokens en la respuesta.

    Returns:
        El texto generado por el modelo.
    """
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_output_tokens
        )
    )
    return response.text

def get_prompt(inputs, outputs):
    mapita = {0: "no superviviente", 1: "superviviente"}
    prompt = f"""Hola Gemini! Long time no see! 
    Estoy haciendo una API de prediccion de superviviente o no con el dataset del titanic. He usado solo 3 features:
    PClass,
    Sex,
    Age.

    Lo que necesito es pasarte los datos de los inputs y la prediccion del modelo y que me generes un breve texto especulando 
    a traves los inputs dados y la prediccion del modelo los motivos por los cuales el modelo ha hecho esa prediccion y si tiene
    sentido o no la prediccion dado el contexto.

    Quiero que lo escribas de forma muy narrada, como si fuera una historia de aventuras. Pero quiero un texto conciso (entre 100 y 500 palabras maximo)

    Se creativo y mójate.

    IMPORTANTE: el formato de salida ha de ser UNICA y EXCLUSIVAMENTE el texto narrado. No me des saludos, metadatos ni nada aparte del breve texto. 
    IMPORTANTE 2: Omite todo tipo de formato enriquecido (markdown, html, etc... ) dame solo texto pano.
    IMPORTANTE 3: Para que el texto no resulte muy horizontal, incluye numerosos saltos de linea.

    EL CONTEXTO ES EL SIGUIENTE:
    INPUTS:
    PClass: {inputs[0]}
    Sex: {inputs[1]} (siendo 0 male y 1 female)
    Age: {inputs[2]} (en años)

    Prediccion: {mapita[outputs]}

    Tu respuesta aqui:
    """
    return prompt