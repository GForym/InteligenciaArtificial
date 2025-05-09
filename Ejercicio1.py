import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configurar las variables de entorno para la caché de modelos
os.environ["TRANSFORMERS_CACHE"] = "./model_cache"
os.environ["HF_HOME"] = "./model_cache"

def cargar_modelo(nombre_modelo):
    """
    Carga un modelo pre-entrenado y su tokenizador correspondiente.
    Optimizado para usar half-precision en GPU y regular en CPU.
    """
    try:
        # Cargar tokenizador con configuración para español
        tokenizador = AutoTokenizer.from_pretrained(
            nombre_modelo,
            padding_side="left",
            use_fast=True
        )
        
        # Configurar pad_token si no existe
        if tokenizador.pad_token is None:
            tokenizador.pad_token = tokenizador.eos_token

        # Determinar el tipo de dato según el dispositivo disponible
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Cargar modelo con configuración optimizada
        modelo = AutoModelForCausalLM.from_pretrained(
            nombre_modelo,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        # Configurar el modelo para inferencia
        modelo.eval()
        
        # Mover a GPU si está disponible
        if torch.cuda.is_available():
            modelo.to("cuda")
            
        return modelo, tokenizador
    
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        raise

def verificar_dispositivo():
    """
    Verifica el dispositivo disponible y muestra información detallada.
    """
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print("\n[+] GPU detectada:")
        print(f"  Nombre: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        dispositivo = torch.device("cpu")
        print("\n[!] No se detectó GPU. Usando CPU.")
        print("  Recomendación: Para mejores resultados, usa una GPU con soporte CUDA")
    
    return dispositivo

def generar_texto(modelo, tokenizador, dispositivo, prompt, max_length=100):
    """
    Genera texto con configuración optimizada para coherencia.
    """
    try:
        # Codificar el prompt
        inputs = tokenizador(
            prompt, 
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(dispositivo)
        
        # Generar texto con parámetros ajustados
        with torch.no_grad():
            outputs = modelo.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                pad_token_id=tokenizador.eos_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        # Decodificar y limpiar el texto generado
        texto_generado = tokenizador.decode(
            outputs[0], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return texto_generado
    
    except Exception as e:
        print(f"Error en generación de texto: {str(e)}")
        return None

def main():
    print("=== Configuración inicial ===")
    dispositivo = verificar_dispositivo()
    
    # Seleccionar modelo (optimizado para español)
    nombre_modelo = "PlanTL-GOB-ES/gpt2-base-bne"  # Modelo GPT-2 en español
    
    print(f"\n=== Cargando modelo: {nombre_modelo} ===")
    try:
        modelo, tokenizador = cargar_modelo(nombre_modelo)
        print("Modelo cargado exitosamente!")
    except:
        print("\n[!] No se pudo cargar el modelo principal. Usando alternativa...")
        nombre_modelo = "bertin-project/bertin-gpt-j-6B"  # Alternativa más ligera
        modelo, tokenizador = cargar_modelo(nombre_modelo)
    
    # Ejemplos de prueba en español
    prompts = [
        "Hola, ¿cómo estás hoy?",
        "El futuro de la inteligencia artificial en España",
        "Para aprender programación en Python,"
    ]
    
    print("\n=== Generando respuestas ===")
    for i, prompt in enumerate(prompts, 1):
        print(f"\nPrompt #{i}: '{prompt}'")
        respuesta = generar_texto(modelo, tokenizador, dispositivo, prompt)
        
        if respuesta:
            print("\nRespuesta generada:")
            print("-" * 50)
            print(respuesta)
            print("-" * 50)
        else:
            print("Error al generar respuesta para este prompt.")

if __name__ == "__main__":
    main()
