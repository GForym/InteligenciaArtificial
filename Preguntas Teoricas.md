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

### 3. Describa las técnicas principales para reducir el problema de "alucinaciones" en chatbots basados en LLMs. ¿Qué estrategias podemos implementar a nivel de inferencia y a nivel de prompt engineering para mejorar la precisión factual de las respuestas?
