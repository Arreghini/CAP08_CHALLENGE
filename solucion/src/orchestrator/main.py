import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv
import requests

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener las API Keys desde las variables de entorno
serper_api_key = os.getenv("SERPER_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

def search_google(query: str):
    """
La función search_google utiliza la API de Serper para buscar en Google según una consulta y devuelve una lista de hasta 5 enlaces de resultados 
de búsqueda. 
query: El parámetro query en la función search_google es una cadena que representa la consulta de búsqueda que deseas realizar en Google utilizando
la API de Serper. Puedes ingresar cualquier consulta de búsqueda que desees buscar en Google.
query: str
:return: Una lista de diccionarios que contiene el título y el enlace de los 5 principales resultados de búsqueda de Google basados en la consulta
proporcionada. Si hay un error durante la búsqueda, se devolverá una lista vacía.
    """
    """Buscar en Google usando la API de Serper"""
    url = "https://google.serper.dev/search"
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "q": query,
        "gl": "ar",  # Argentina
        "hl": "es"   # Idioma Español
    })

    response = requests.post(url, headers=headers, data=data)
    if response.status_code == 200:
        search_results = response.json()
        links = [{"title": result['title'], "link": result['link']} for result in search_results.get('organic', [])][:5]
        return links
    else:
        print(f"Error en la búsqueda: {response.status_code} - {response.text}")
        return []
def extract_text_from_url(url: str) -> str:
    """
    Extraer el texto principal de una página web
    Parameters:
        url (str): La URL de la página web de la que se extraerá el texto.
    Returns:
        str: El texto principal extraído de la página web, o un mensaje de error si no se pudo extraer contenido relevante.
    """
    if not url:
        return "No se pudo extraer contenido relevante."  # Manejo de URL vacía
    try:
        response = requests.get(url)
        response.raise_for_status()

        # Parsear el contenido HTML de la página
        soup = BeautifulSoup(response.text, 'html.parser')

        # Intentar extraer el texto de los párrafos
        paragraphs = soup.find_all('p')
        article_text = "\n".join([para.get_text() for para in paragraphs])

        # Si no hay párrafos, intentar extraer todo el texto visible
        if not article_text.strip():
            article_text = "\n".join(soup.stripped_strings)

        # Devolver el texto extraído
        return article_text.strip() if article_text else "No se pudo extraer contenido relevante."
    
    except requests.RequestException as e:
        print(f"Error al acceder a la URL {url}: {str(e)}")
        return "Error al extraer el contenido."

def extract_texts_from_search_results(query: str):
    """
Buscar en Google y extraer el texto de los primeros 5 resultados
query: La consulta de búsqueda para buscar en Google y extraer texto de los resultados.
query: str
:return: Una lista de diccionarios con el título, enlace y contenido de los primeros 5 resultados de búsqueda de Google.
    """
    """Buscar en Google y extraer el texto de los primeros 5 resultados"""
    search_results = search_google(query)

    if not search_results:
        return "No se encontraron resultados relevantes."

    extracted_texts = []
    for result in search_results:
        print(f"Extrayendo contenido de: {result['link']}")
        text = extract_text_from_url(result['link'])
        extracted_texts.append({
            "title": result['title'],
            "link": result['link'],
            "content": text
        })
    
    return extracted_texts

def interact_with_llm_huggingface_streaming(user_input: str, extracted_texts: list):
    """
Interactuar con el modelo de lenguaje utilizando Hugging Face con streaming.
Parameters:
user_input (str): La pregunta o solicitud del usuario.
extracted_texts (list): Una lista de diccionarios que contiene los datos de texto extraídos con las claves 'title', 'link', y 'content'.
Returns:
    None
Raises:
    None
    """
    """Interactuar con el modelo de lenguaje utilizando Hugging Face con streaming"""
    
    extracted_info = "\n\n".join([f"Título: {item['title']}\nEnlace: {item['link']}\nContenido:\n{item['content']}" for item in extracted_texts])
    
    # Crear el texto completo que le enviaremos al modelo
    prompt = f"Información extraída:\n{extracted_info}\n\nPregunta del usuario: {user_input}\n\nGenera una respuesta basada en la información anterior."

    api_url = "https://api-inference.huggingface.co/models/gpt2"  # Puedes cambiar el modelo según tus necesidades
    headers = {
        "Authorization": f"Bearer {huggingface_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": prompt,
        "stream": True,  # Habilitar el streaming
        "parameters": {
            "max_new_tokens": 500,  # Ajusta la cantidad de tokens a generar
            "temperature": 0.7
        }
    }

    # Realizar la solicitud con streaming
    response = requests.post(api_url, headers=headers, json=data, stream=True)

    if response.status_code != 200:
        print(f"Error en la solicitud: {response.status_code}")
        return

    # Leer las respuestas del streaming a medida que se generan
    print("Generando respuesta en tiempo real:\n")
    for line in response.iter_lines():
        if line:
        
            response_text = line
        print(response_text)

if __name__ == "__main__":
    # Consulta de ejemplo
    query = input("Ingrese su consulta: ")
    extracted_texts = extract_texts_from_search_results(query)

    # Mostrar los textos extraídos
    for i, text_data in enumerate(extracted_texts, 1):
        print(f"\nResultado {i}: {text_data['title']} ({text_data['link']})")
        print(f"Contenido extraído:\n{text_data['content'][:1000]}...")  # Limitar a los primeros 1000 caracteres

    # Interactuar con el modelo de Hugging Face utilizando streaming
    user_input = input("\nHaz una pregunta basada en la información extraída: ")
    interact_with_llm_huggingface_streaming(user_input, extracted_texts)
