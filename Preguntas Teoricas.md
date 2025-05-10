# Presentado Por Gustavo Fory 
# Preguntas Teoricas 

### 1. ¿Cuáles son las diferencias fundamentales entre los modelos encoder-only, decoder-only y encoder-decoder en el contexto de los chatbots conversacionales? Explique qué tipo de modelo sería más adecuado para cada caso de uso y por qué.
  a) Encoder-only (Ej: BERT):
    Estos modelos están diseñados principalmente para comprensión de texto.
    Funcionan procesando toda la entrada simultáneamente, lo que les permite generar representaciones profundas del texto.
    No están diseñados para generación de texto, sino para tareas como clasificación, análisis de sentimiento, QA (pregunta-respuesta extractiva), etc.

Uso adecuado:
  Chatbots que solo necesitan comprender el texto del usuario y responder usando plantillas predefinidas o consultas a base de datos.
  Ejemplo: sistemas de atención con respuestas cerradas o comandos específicos.

b) Decoder-only (Ej: GPT):
Están optimizados para generación de texto, prediciendo la siguiente palabra dada la secuencia anterior (autoregresivos).
No tienen una etapa explícita de codificación del input separado del output.
Uso adecuado:
Chatbots conversacionales generativos como ChatGPT, donde la tarea principal es producir respuestas coherentes, fluidas y variadas.
Útil para diálogos abiertos, redacción creativa o asistentes virtuales avanzados.

c) Encoder-Decoder (Ej: T5, BART):
Combinan comprensión (encoder) y generación (decoder).
El encoder procesa la entrada completa, y el decoder genera la salida con atención cruzada a la representación del encoder.
Uso adecuado:
Chatbots que requieren transformar o traducir el input, como en tareas de respuesta compleja, resumen, o traducción multilingüe.
También útiles cuando se necesita un control más claro sobre la entrada/salida (por ejemplo, QA generativa con contexto controlado).

### 2. Explique el concepto de "temperatura" en la generación de texto con LLMs. ¿Cómo afecta al comportamiento del chatbot y qué consideraciones debemos tener al ajustar este parámetro para diferentes aplicaciones?
La temperatura en los modelos de lenguaje (LLMs) es un parámetro que controla el grado de aleatoriedad en la generación de texto. En términos simples, determina cuán "creativo" o "conservador" es el modelo al elegir la próxima palabra en una secuencia. Su valor usualmente oscila entre 0 y 1 (aunque puede superar 1), donde valores bajos como 0 o 0.2 hacen que el modelo sea más determinista y repetitivo, eligiendo palabras con mayor probabilidad, mientras que valores altos como 0.8 o 1.0 aumentan la variedad y originalidad, aunque con mayor riesgo de incoherencia.
El comportamiento del chatbot cambia significativamente según este parámetro. A temperatura baja, es ideal para tareas que requieren precisión, como programación, matemáticas o respuestas técnicas, ya que produce resultados más predecibles. A temperatura alta, es más útil para tareas creativas como generación de historias, poesía o brainstorming, donde la diversidad y novedad son deseables.

### 3. Describa las técnicas principales para reducir el problema de "alucinaciones" en chatbots basados en LLMs. ¿Qué estrategias podemos implementar a nivel de inferencia y a nivel de prompt engineering para mejorar la precisión factual de las respuestas?
Las "alucinaciones" en LLMs se refieren a respuestas que suenan plausibles pero son incorrectas o inventadas. Para reducir este problema, se emplean varias técnicas tanto a nivel de inferencia como de prompt engineering que buscan mejorar la precisión factual del modelo.
A nivel de inferencia, una estrategia común es el uso de modelos con acceso a herramientas externas, como motores de búsqueda o bases de datos, lo que permite verificar hechos en tiempo real. También se puede aplicar la verificación de consistencia, pidiendo al modelo generar varias respuestas y comparar su coherencia. Otra técnica es el post-procesamiento, donde se validan las respuestas usando reglas o modelos especializados para detectar errores factuales.
En cuanto al prompt engineering, una buena práctica es usar instrucciones claras y específicas, que orienten al modelo a responder de forma más precisa. También se puede incorporar contexto adicional confiable dentro del prompt (como fragmentos de textos verificados) o utilizar prompts encadenados, donde una primera respuesta es revisada o refinada en una segunda etapa.
En conjunto, estas técnicas ayudan a reducir errores y mejorar la confianza en la información generada, especialmente en aplicaciones críticas como educación, medicina o derecho.
