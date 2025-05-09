##Instalar 
!pip install gradio
!pip install --upgrade torch torchvision torchaudio
!pip install --upgrade transformers

!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

!pip install transformers
## estrucutra codigo 
import gradio as gr
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from huggingface_hub import login

def configurar_peft(modelo, r=8, lora_alpha=32, lora_dropout=0.1):
    """
    Configura el modelo para fine-tuning con PEFT/LoRA.

    Args:
        modelo: Modelo base
        r (int): Rango de adaptadores LoRA
        lora_alpha (int): Escala alpha para LoRA
        lora_dropout (float): Dropout probability de LoRA

    Returns:
        modelo: Modelo adaptado para fine-tuning
    """
    config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    modelo_peft = get_peft_model(modelo, config)
    modelo_peft.print_trainable_parameters()
    return modelo_peft

def guardar_modelo(modelo, tokenizador, ruta):
    """
    Guarda el modelo y tokenizador en una ruta específica.

    Args:
        modelo: Modelo a guardar
        tokenizador: Tokenizador del modelo
        ruta (str): Ruta donde guardar
    """
    modelo.save_pretrained(ruta)
    tokenizador.save_pretrained(ruta)
    print(f"Modelo y tokenizador guardados en: {ruta}")

def cargar_modelo_personalizado(ruta):
    """
    Carga un modelo personalizado desde una ruta específica.

    Args:
        ruta (str): Ruta del modelo

    Returns:
        tuple: (modelo, tokenizador)
    """
    # Verificar si la ruta existe
    if os.path.exists(ruta):
        modelo = AutoModelForCausalLM.from_pretrained(ruta)
        tokenizador = AutoTokenizer.from_pretrained(ruta)
        print(f"Modelo y tokenizador cargados desde: {ruta}")
        return modelo, tokenizador
    else:
        print(f"Error: La ruta '{ruta}' no existe o no es válida.")
        return None, None

def autenticar_huggingface():
    """
    Realiza la autenticación en Hugging Face si es necesario.
    """
    try:
        login()  # Intentar autenticar si no lo está
        print("Autenticado correctamente en Hugging Face.")
    except Exception as e:
        print(f"Error al autenticar: {e}")

# Interfaz web simple con Gradio
def crear_interfaz_web(chatbot_pipeline):
    """
    Crea una interfaz web simple para el chatbot usando Gradio.

    Args:
        chatbot_pipeline: Pipeline de Transformers para el chatbot

    Returns:
        gr.Interface: Interfaz de Gradio
    """
    def responder(mensaje, historial=[]):
        respuesta = chatbot_pipeline(mensaje, conversation_id=historial)[0]['generated_text']
        historial.append((mensaje, respuesta))
        return "", historial

    inputs = [gr.Textbox(label="Mensaje"), gr.Chatbot(label="Chatbot")]
    outputs = [gr.Textbox(), gr.Chatbot()]

    interfaz = gr.Interface(fn=responder,
                            inputs=inputs,
                            outputs=outputs,
                            title="Chatbot Personalizado",
                            description="Interactúa con el chatbot personalizado.")
    return interfaz

# Función principal para el despliegue
def main_despliegue():
    ruta_modelo_personalizado = "mi_modelo_personalizado"  # Cambia esto a la ruta real de tu modelo

    # Si estás usando Hugging Face para modelos privados, autentícate
    autenticar_huggingface()

    # Intentar cargar el modelo personalizado desde la ruta local
    modelo, tokenizador = cargar_modelo_personalizado(ruta_modelo_personalizado)

    if modelo is None or tokenizador is None:
        print("Intentando cargar modelo desde Hugging Face...")
        try:
            # Si el modelo no existe localmente, cargar desde Hugging Face
            modelo = AutoModelForCausalLM.from_pretrained("username/mi_modelo_personalizado")  # Cambia "username" por el tuyo
            tokenizador = AutoTokenizer.from_pretrained("username/mi_modelo_personalizado")  # Cambia "username" por el tuyo
            print("Modelo y tokenizador cargados desde Hugging Face.")
        except Exception as e:
            print(f"Error al cargar el modelo desde Hugging Face: {e}")
            return

    # Crear instancia del pipeline de chatbot
    chatbot_pipeline = pipeline("conversational", model=modelo, tokenizer=tokenizador)

    # Crear y lanzar la interfaz web
    interfaz = crear_interfaz_web(chatbot_pipeline)
    interfaz.launch(share=False)  # share=True para compartir públicamente (con precaución)

if __name__ == "__main__":
    main_despliegue()


