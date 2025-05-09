from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import time
import torch.nn as nn
import gc

def verificar_cuantizacion_disponible():
    """Verifica si la cuantización con bitsandbytes está disponible."""
    try:
        import bitsandbytes
        try:
            bitsandbytes.lib.libbitsandbytes.get_cuda_version()
            return True
        except AttributeError:
            return False
    except ImportError:
        return False

def configurar_cuantizacion(bits=4):
    """
    Configura los parámetros para la cuantización del modelo.
    Args:
        bits (int): Bits para cuantización (4 u 8)
    Returns:
        BitsAndBytesConfig: Configuración de cuantización o None si no está disponible.
    """
    if not verificar_cuantizacion_disponible():
        print("Advertencia: bitsandbytes no está configurado correctamente para GPU, la cuantización no estará disponible.")
        return None
    if bits not in [4, 8]:
        raise ValueError("El número de bits debe ser 4 u 8.")
    try:
        config_cuantizacion = BitsAndBytesConfig(
            load_in_4bit=True if bits == 4 else False,
            load_in_8bit=True if bits == 8 else False,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True, # Permite offloading a CPU si es necesario
        )
        return config_cuantizacion
    except ImportError as e:
        print(f"Error al configurar la cuantización: {e}")
        return None

def cargar_modelo_optimizado(nombre_modelo, optimizaciones=None):
    """
    Carga un modelo con optimizaciones aplicadas.
    Args:
        nombre_modelo (str): Identificador del modelo
        optimizaciones (dict): Diccionario con flags para las optimizaciones
    Returns:
        tuple: (modelo, tokenizador)
    """
    if optimizaciones is None:
        optimizaciones = {
            "cuantizacion": True,
            "bits": 4,
            "offload_cpu": False,
            "flash_attention": True
        }

    tokenizer = AutoTokenizer.from_pretrained(nombre_modelo)
    quantization_config = None
    if optimizaciones.get("cuantizacion", False) and verificar_cuantizacion_disponible():
        quantization_config = configurar_cuantizacion(bits=optimizaciones.get("bits", 4))
        if quantization_config is None:
            print("Advertencia: La configuración de cuantización no se aplicó.")
    elif optimizaciones.get("cuantizacion", False) and not verificar_cuantizacion_disponible():
        print("Advertencia: Se solicitó cuantización, pero bitsandbytes no está disponible para GPU. Se cargará el modelo sin cuantización.")

    try:
        attn_implementation = "flash_attention_2" if optimizaciones.get("flash_attention", False) else "eager"
    except ImportError:
        print("Advertencia: Flash Attention no está instalado. Se utilizará la implementación eager.")
        attn_implementation = "eager"

    modelo = AutoModelForCausalLM.from_pretrained(
        nombre_modelo,
        quantization_config=quantization_config,
        torch_dtype=torch.float16 if quantization_config else None,
        low_cpu_mem_usage=True, # Ayuda a reducir el uso de memoria durante la carga
        device_map="auto" if not optimizaciones.get("offload_cpu", False) else "cpu",
        attn_implementation=attn_implementation
    )

    return modelo, tokenizer

def aplicar_sliding_window(modelo, window_size=1024):
    """
    Configura la atención de ventana deslizante para procesar secuencias largas.
    Args:
        modelo: Modelo a configurar
        window_size (int): Tamaño de la ventana de atención
    """
    config = modelo.config
    for name, module in modelo.named_modules():
        if "attn" in name:
            if hasattr(module, 'sliding_window'):
                module.sliding_window = window_size
                if hasattr(module, 'config'):
                    module.config.sliding_window = window_size
            elif hasattr(module, 'self_attn') and hasattr(module.self_attn, 'sliding_window'):
                module.self_attn.sliding_window = window_size
                if hasattr(module.self_attn, 'config'):
                    module.self_attn.config.sliding_window = window_size
            elif "GPT2Attention" in module.__class__.__name__:
                print(f"Nota: La configuración de sliding window para {name} ({module.__class__.__name__}) puede requerir una implementación específica.")

def evaluar_rendimiento(modelo, tokenizador, texto_prueba, dispositivo):
    """
    Evalúa el rendimiento del modelo en términos de velocidad y memoria.
    Args:
        modelo: Modelo a evaluar
        tokenizador: Tokenizador del modelo
        texto_prueba (str): Texto para pruebas de rendimiento
        dispositivo: Dispositivo donde se ejecutará
    Returns:
        dict: Métricas de rendimiento
    """
    modelo.eval()
    inputs = tokenizador(texto_prueba, return_tensors="pt").to(dispositivo)
    with torch.no_grad():
        start_time = time.time()
        outputs = modelo.generate(**inputs, max_length=200)
        end_time = time.time()
    generated_text = tokenizador.decode(outputs[0], skip_special_tokens=True)
    inference_time = end_time - start_time
    num_tokens_generated = len(tokenizador.encode(generated_text))
    tokens_per_second = num_tokens_generated / inference_time if inference_time > 0 else 0

    memory_usage = torch.cuda.max_memory_allocated(device=dispositivo) / (1024 ** 2) if dispositivo == "cuda" and torch.cuda.is_available() else "N/A (CPU)"

    return {
        "inference_time": f"{inference_time:.4f} segundos",
        "memory_usage": f"{memory_usage}",
        "tokens_per_second": f"{tokens_per_second:.2f}",
        "generated_text": generated_text
    }

def demo_optimizaciones(nombre_modelo="gpt2"):
    """
    Crea y evalúa diferentes configuraciones.
    1. Modelo base sin optimizaciones
    2. Modelo con cuantización de 4 bits
    3. Modelo con sliding window attention
    4. Modelo con todas las optimizaciones
    """
    texto_prueba = "La inteligencia artificial es"
    dispositivo = "cpu" # Mantener en CPU para tu entorno actual

    resultados = {}

    # 1. Modelo base sin optimizaciones
    print("\nEvaluando modelo base sin optimizaciones...")
    modelo_base, tokenizador_base = cargar_modelo_optimizado(nombre_modelo, optimizaciones={"cuantizacion": False, "flash_attention": False})
    modelo_base.to(dispositivo)
    metricas_base = evaluar_rendimiento(modelo_base, tokenizador_base, texto_prueba, dispositivo)
    resultados["base"] = metricas_base
    del modelo_base
    del tokenizador_base
    torch.cuda.empty_cache()
    gc.collect()

    # 2. Modelo con intento de cuantización de 4 bits
    print("\nEvaluando modelo con intento de cuantización de 4 bits...")
    modelo_cuantizado, tokenizador_cuantizado = cargar_modelo_optimizado(nombre_modelo, optimizaciones={"cuantizacion": True, "bits": 4, "flash_attention": False})
    modelo_cuantizado.to(dispositivo)
    metricas_cuantizado = evaluar_rendimiento(modelo_cuantizado, tokenizador_cuantizado, texto_prueba, dispositivo)
    resultados["cuantizado_4bit"] = metricas_cuantizado
    del modelo_cuantizado
    del tokenizador_cuantizado
    torch.cuda.empty_cache()
    gc.collect()

    # 3. Modelo con sliding window attention
    print("\nEvaluando modelo con sliding window attention...")
    modelo_sw, tokenizador_sw = cargar_modelo_optimizado(nombre_modelo, optimizaciones={"cuantizacion": False, "flash_attention": False})
    aplicar_sliding_window(modelo_sw)
    modelo_sw.to(dispositivo)
    metricas_sw = evaluar_rendimiento(modelo_sw, tokenizador_sw, texto_prueba, dispositivo)
    resultados["sliding_window"] = metricas_sw
    del modelo_sw
    del tokenizador_sw
    torch.cuda.empty_cache()
    gc.collect()

    # 4. Modelo con todas las optimizaciones (sin Flash Attention en CPU)
    print("\nEvaluando modelo con intento de cuantización de 4 bits y sliding window attention...")
    modelo_combinado, tokenizador_combinado = cargar_modelo_optimizado(nombre_modelo, optimizaciones={"cuantizacion": True, "bits": 4, "flash_attention": False})
    aplicar_sliding_window(modelo_combinado)
    modelo_combinado.to(dispositivo)
    metricas_combinado = evaluar_rendimiento(modelo_combinado, tokenizador_combinado, texto_prueba, dispositivo)
    resultados["combinado"] = metricas_combinado
    del modelo_combinado
    del tokenizador_combinado
    torch.cuda.empty_cache()
    gc.collect()

    print("\n--- Resultados de la Evaluación ---")
    for nombre, metricas in resultados.items():
        print(f"\nConfiguración: {nombre}")
        for clave, valor in metricas.items():
            print(f"- {clave}: {valor}")

    print("\nNota sobre las optimizaciones en CPU:")
    print("- La cuantización con bitsandbytes se ejecuta en GPU. En CPU, la carga del modelo se realizará sin cuantización.")
    print("- Flash Attention está diseñado para GPUs y no tendrá un efecto en CPU.")
    print("- La aplicación de sliding window attention depende de la arquitectura del modelo y puede no ser efectiva para gpt2.")
    print("Para una evaluación precisa de estas optimizaciones, se recomienda un entorno con GPU.")

if __name__ == "__main__":
    demo_optimizaciones()
