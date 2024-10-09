import requests
from bs4 import BeautifulSoup
import json
import os
from dotenv import load_dotenv

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener las API Keys desde las variables de entorno
serper_api_key = os.getenv("SERPER_API_KEY")
huggingface_api_key = os.getenv("HUGGING_FACE_API_KEY")

def search_google(query: str):
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
    if not url:
        return "No se pudo extraer contenido relevante."
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        article_text = "\n".join([para.get_text() for para in paragraphs])
        if not article_text.strip():
            article_text = "\n".join(soup.stripped_strings)
        return article_text.strip() if article_text else "No se pudo extraer contenido relevante."
    
    except requests.RequestException as e:
        print(f"Error al acceder a la URL {url}: {str(e)}")
        return "Error al extraer el contenido."

def extract_texts_from_search_results(query: str):
    search_results = search_google(query)

    if not search_results:
        return []

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
    if not user_input.strip():
        print("La entrada del usuario está vacía. No se realizará la solicitud.")
        return None

    extracted_info = "\n\n".join([f"Título: {item['title']}\nEnlace: {item['link']}\nContenido:\n{item.get('content', 'Contenido no disponible.')}" for item in extracted_texts])

    prompt = f"Información extraída:\n{extracted_info}\n\nPregunta del usuario: {user_input}\n\nGenera una respuesta basada en la información anterior."
    
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B"
    
    headers = {
        "Authorization": f"Bearer {huggingface_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": prompt,
        "stream": True,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7
        }
    }

    response = requests.post(api_url, headers=headers, json=data, stream=True)

    if response.status_code != 200:
        print(f"Error en la solicitud: {response.status_code}")
        return None

    print("Generando respuesta en tiempo real:\n")
    generated_response = ""
    for line in response.iter_lines():
        if line:  # Verificar si la línea no está vacía
            print(line.decode('utf-8'))  # Imprimir la línea decodificada
            generated_response += line.decode('utf-8') + "\n"  # Agregar la línea directamente

    return generated_response

if __name__ == "__main__":
    query = input("Ingrese su consulta: ")
    extracted_texts = extract_texts_from_search_results(query)

    # Mostrar los textos extraídos
    for i, text_data in enumerate(extracted_texts, 1):
        print(f"\nResultado {i}: {text_data['title']} ({text_data['link']})")
        print(f"Contenido extraído:\n{text_data['content'][:1000]}...")  # Limitar a los primeros 1000 caracteres

    user_input = input("\nHaz una pregunta basada en la información extraída: ")
    response = interact_with_llm_huggingface_streaming(user_input, extracted_texts)
    
    if response:
        print("\nRespuesta generada por el modelo:\n", response)
