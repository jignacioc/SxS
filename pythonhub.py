import requests
from faster_whisper import WhisperModel
from inputvoice import capture_audio
from rag_system import RAGSystem
from sentence_transformers import SentenceTransformer, util
from config import MISTRAL_API_KEY, MISTRAL_API_URL

# Ruta del archivo de audio generado por inputvoice.py
audio_file = "output.wav"

# Función para convertir voz a texto usando Faster Whisper
def voice_to_text(audio_path):
    model = WhisperModel("small")  # Inicializar el modelo Faster Whisper
    segments, info = model.transcribe(audio_path)
    # Unir los segmentos en un solo texto
    text = ' '.join([segment.text for segment in segments])
    return text

# Función para generar respuesta con Mistral AI
def generate_response_with_mistral(text, context):
    headers = {
        'Authorization': f'Bearer {MISTRAL_API_KEY}',
        'Content-Type': 'application/json'
    }
    messages = [
        {'role': 'system', 'content': 'Eres un asistente útil.'},
        {'role': 'user', 'content': text}
    ]
    
    if context:
        messages.append({'role': 'system', 'content': f"Información relevante: {context}"})
    
    payload = {
        'model': 'mistral-small-latest',
        'messages': messages,
        'temperature': 0.7,
        'top_p': 1,
        'max_tokens': 512,
        'stream': False,
        'safe_prompt': False,
        'random_seed': 1337
    }
    
    try:
        response = requests.post(MISTRAL_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        return response.json().get('choices', [{}])[0].get('message', {}).get('content')
    except requests.exceptions.RequestException as e:
        print(f"Error al comunicarse con Mistral AI: {e}")
        return None

# Función principal del hub
def python_hub():
    # Inicializar sistema RAG
    rag = RAGSystem()
    
    # Indexar documentos de ejemplo (esto debería ser reemplazado por tu corpus)
    docs = [
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "La inteligencia artificial está transformando el mundo.",
        "Python es un lenguaje de programación muy popular."
    ]
    rag.index_documents(docs)

    # Inicializar el modelo de embeddings para calcular similitud
    similarity_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    while True:
        # Capturar audio
        capture_audio()

        # Convertir voz a texto
        text = voice_to_text(audio_file)
        print(f"Texto reconocido: {text}")

        # Recuperar información relevante utilizando RAG
        results = rag.retrieve(text)
        
        # Filtrar contexto relevante
        relevant_context = [docs[i] for i in results if i >= 0]
        
        # Calcular la similitud entre el texto del usuario y los contextos recuperados
        query_embedding = similarity_model.encode(text, convert_to_tensor=True)
        context_embeddings = similarity_model.encode(relevant_context, convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(query_embedding, context_embeddings)
        
        # Umbral de relevancia: sólo consideramos contexto si hay suficiente relevancia
        context_threshold = 0.5  # Este umbral puede ajustarse según la necesidad
        if cosine_scores.max() > context_threshold:
            context = ". ".join(relevant_context)
        else:
            context = ""
        
        print(f"Contexto relevante: {context}")

        # Generar respuesta con Mistral AI
        response = generate_response_with_mistral(text, context)
        if response:
            print(f"Respuesta generada: {response}")

if __name__ == "__main__":
    python_hub()
