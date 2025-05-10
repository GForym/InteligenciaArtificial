
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import List, Dict, Optional

class ContextManager:
    def __init__(self, max_context: int = 1024):
        self.history: List[Dict[str, str]] = []
        self.max_context = max_context

    def add_message(self, role: str, content: str) -> None:
        """Adds a message to the conversation history."""
        if not content.strip():
            raise ValueError("Message content cannot be empty")
        self.history.append({"role": role, "content": content})

    def get_context(self) -> List[Dict[str, str]]:
        """Returns a copy of the conversation history."""
        return self.history.copy()

    def truncate_context(self, tokenizer: AutoTokenizer) -> None:
        """Truncates history if it exceeds token limit."""
        if not self.history:
            return

        # Convertir todo el historial en texto para la tokenización
        context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
        tokens = tokenizer.tokenize(context_text)

        while len(tokens) > self.max_context and len(self.history) > 1:
            # Conservar el primer mensaje y eliminar los más antiguos
            self.history = [self.history[0]] + self.history[2:]
            context_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.history])
            tokens = tokenizer.tokenize(context_text)

    def clear_context(self) -> None:
        """Clears conversation history while keeping system instructions."""
        if self.history and self.history[0]["role"] == "system":
            system_msg = self.history[0]
            self.history = [system_msg]
        else:
            self.history = []


class Chatbot:
    def __init__(self, model_id: str = "microsoft/DialoGPT-medium"):
        # Configuracion del Modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id)

        # Configuración del dispositivo
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # Proceso de generacion de texto
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device
        )

        # Gestion del contexto
        self.context_manager = ContextManager()

        # Instrucciones del sistema
        self._initialize_instructions()

    def _initialize_instructions(self) -> None:
        """Initializes system instructions for the chatbot."""
        instructions = (
            "Eres un guía turístico entusiasta y conocedor de Cali, Colombia. "
            "Respondes de manera amigable y apasionada sobre los lugares de interés, la cultura y la gastronomía de la ciudad. "
            "Anima a los visitantes a explorar Cali y comparte datos interesantes."
        )
        self.context_manager.add_message("system", instructions)

    def respond(self, user_message: str) -> str:
        """Generates a response to the user's message."""
        # Validacion de las entradas
        if not user_message.strip():
            return "Please send a message with actual content."

        try:
            # Añadir mensaje de usuario al contexto
            self.context_manager.add_message("user", user_message)

            # Truncar el contexto si es necesario
            self.context_manager.truncate_context(self.tokenizer)

            # Preparar los mensajes para el modelo
            messages = self.context_manager.get_context()

            # Generar respuesta mediante pipeline
            response = self.generator(
                self.tokenizer.apply_chat_template(messages, tokenize=False),
                max_new_tokens=200,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True,
                return_full_text=False
            )[0]['generated_text']

            # Limpiar la respuesta
            response = self._clean_response(response)

            # Añadir respuesta al contexto
            self.context_manager.add_message("assistant", response)

            return response

        except Exception as e:
            return f"An error occurred while generating the response: {str(e)}"

    def _clean_response(self, response: str) -> str:
        """Cleans up the model's response by removing unwanted content."""
        for msg in self.context_manager.history:
            if msg['content'] in response:
                response = response.replace(msg['content'], '')

        # Elimine los espacios en blanco y las nuevas líneas iniciales y finales.
        response = response.strip()

        # Eliminar cualquier texto después de una posible segunda línea nueva
        response = response.split('\n')[0]

        return response

    def reset_conversation(self) -> None:
        """Resets the conversation while keeping system instructions."""
        self.context_manager.clear_context()


def conversation_test() -> None:
    """Test function for chatbot conversation as a Cali tour guide."""
    chatbot = Chatbot()

    # Modificamos las instrucciones del sistema para este rol
    guia_instrucciones = (
        "Eres un guía turístico entusiasta y conocedor de Cali, Colombia. "
        "Respondes de manera amigable y apasionada sobre los lugares de interés, la cultura y la gastronomía de la ciudad. "
        "Anima a los visitantes a explorar Cali y comparte datos interesantes."
    )
    chatbot.context_manager.history[0]["content"] = guia_instrucciones
    chatbot.context_manager.clear_context() # Limpiamos y volvemos a agregar solo las instrucciones

    messages = [
        "¡Hola! Estoy planeando visitar Cali, ¿qué me recomiendas ver primero?",
        "He oído hablar de la salsa en Cali, ¿dónde puedo experimentarla?",
        "¿Cuál es la historia del Cristo Rey?",
        "¿Qué plato típico de Cali debo probar?",
        "¡Muchas gracias por tus recomendaciones!"
    ]

    for message in messages:
        print(f"Usuario: {message}")
        response = chatbot.respond(message)
        print(f"Asistente (Guía de Cali): {response}\n")

    # Reiniciar conversación (opcional, para futuras pruebas)
    chatbot.reset_conversation()
    print("Conversación reiniciada")

if __name__ == "__main__":
    conversation_test()
