import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configurar caché (opcional pero útil en Colab)
os.environ['TRANSFORMERS_CACHE'] = '/content/transformers_cache'
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

def cargar_modelo(nombre_modelo):
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo)
    if torch.cuda.is_available():
        modelo.half()
    modelo.eval()
    return modelo, tokenizador

def verificar_dispositivo():
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        dispositivo = torch.device("cpu")
        print("GPU no disponible, usando CPU.")
    return dispositivo

def preprocesar_entrada(texto, tokenizador, longitud_maxima=512):
    tokens = tokenizador.encode_plus(
        texto,
        max_length=longitud_maxima,
        truncation=True,
        return_tensors="pt"
    )
    return tokens['input_ids'].to(dispositivo)

def generar_respuesta(modelo, entrada_procesada, tokenizador, parametros_generacion=None):
    if parametros_generacion is None:
        parametros_generacion = {
            "max_length": 100,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "temperature": 0.7,
            "repetition_penalty": 1.1,
            "pad_token_id": tokenizador.eos_token_id
        }

    salida = modelo.generate(entrada_procesada, **parametros_generacion)
    respuesta = tokenizador.decode(salida[0], skip_special_tokens=True)
    return respuesta

def crear_prompt_sistema(instrucciones):
    return f"[Sistema]: {instrucciones}\n[Usuario]: "

def interaccion_simple():
    global dispositivo
    dispositivo = verificar_dispositivo()
    modelo, tokenizador = cargar_modelo("microsoft/DialoGPT-small")
    modelo.to(dispositivo)

    # Simular historial: usuario y bot
    historial = [
        "Hello!",                           # Turno 1 (Usuario)
        "Hi! How can I assist you today?", # Turno 1 (Bot)
        "Can you tell me a joke?"          # Turno 2 (Usuario)
    ]

    # Construir secuencia de entrada
    input_text = ""
    for i, linea in enumerate(historial):
        input_text += linea + tokenizador.eos_token

    entrada_ids = tokenizador.encode(input_text, return_tensors='pt').to(dispositivo)

    # Generar respuesta
    salida = modelo.generate(
        entrada_ids,
        max_length=entrada_ids.shape[1] + 50,
        pad_token_id=tokenizador.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decodificar solo la nueva respuesta del modelo
    respuesta_generada = tokenizador.decode(salida[0][entrada_ids.shape[-1]:], skip_special_tokens=True)

    print("Entrada del usuario:", historial[-1])
    print("Respuesta generada:", respuesta_generada)
# Ejecutar
interaccion_simple()
