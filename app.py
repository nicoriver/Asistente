import streamlit as st
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from woocommerce import API
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv

# Cargar las variables desde el archivo .env
load_dotenv()

# Configuración de claves
CHAT_OPENAI_KEY = os.getenv("CHAT_OPENAI_KEY")
WC_API_KEY = os.getenv("WC_API_KEY")
WC_API_SECRET = os.getenv("WC_API_SECRET")

# Configuración de WooCommerce
wc = API(
    url="https://desarrollosaltouruguay.com.ar/cosmeticosrosana/",
    consumer_key=WC_API_KEY,
    consumer_secret=WC_API_SECRET,
    version="wc/v3"
)

# Paso 1: Obtener productos desde WooCommerce
def fetch_products():
    response = wc.get("products")
    if response.status_code == 200:
        products = response.json()
        product_descriptions = []
        for product in products:
            name = product.get('name', 'N/A')
            price = product.get('price', 'N/A')
            description = BeautifulSoup(product.get('description', ''), "html.parser").get_text()  
            categories = [cat['name'] for cat in product.get('categories', [])]
            categories_str = ', '.join(categories) if categories else 'Sin categorías'            
           
            product_descriptions.append(
                f"Producto: {name}\nPrecio: {price}\nDescripción: {description}\nCategorías: {categories_str}"
            )
        return product_descriptions
    else:
        return []

# Paso 2: Crear embeddings y vectorstore
product_data = fetch_products()
docs = [Document(page_content=product) for product in product_data]

embeddings = OpenAIEmbeddings(openai_api_key=CHAT_OPENAI_KEY)
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()

# Paso 3: Crear modelo de lenguaje
llm = ChatOpenAI(model="gpt-4", api_key=CHAT_OPENAI_KEY, temperature=0)

# Plantillas de respuesta
RESPONSE_TEMPLATES = {
    "price_query": "El producto '{name}' tiene un precio de {price}.",
    "product_query": (
        "Aquí tienes información sobre el producto:\n"
        "Nombre: {name}\n"
        "Descripción: {description}\n"
        "Categorías: {categories}"
    ),
    "general_help": "Puedes preguntarme sobre productos, precios o información general. ¡Estoy aquí para ayudarte!",
    "general": "Esto es lo que encontré relacionado con tu consulta:\n{response}",
}

# Función para formatear la respuesta
def respuesta(intention, response_data):
    """Formatea la respuesta basada en la intención y datos obtenidos."""
    if intention in ["price_query", "product_query"]:
     
        data = {
            "name": response_data.get("name", "N/A"),
            "price": response_data.get("price", "N/A"),
            "description": response_data.get("description", "N/A"),
            "categories": response_data.get("categories", "Sin categorías"),
        }
        return RESPONSE_TEMPLATES[intention].format(**data)
    elif intention in ["general_help", "general"]:
        return RESPONSE_TEMPLATES[intention].format(response=response_data)
    else:
        return "Lo siento, no entendí tu consulta."

# función respuesta_embeddings para estructurar datos
def respuesta_embeddings(question):
    """Función para buscar en los embeddings y devolver datos estructurados."""
    results = retriever.get_relevant_documents(question)
    if results:
     
        doc = results[0]
      
        content = doc.page_content.split("\n")
        data = {}
        for line in content:
            if line.startswith("Producto:"):
                data["name"] = line.replace("Producto:", "").strip()
            elif line.startswith("Precio:"):
                data["price"] = line.replace("Precio:", "").strip()
            elif line.startswith("Descripción:"):
                data["description"] = line.replace("Descripción:", "").strip()
            elif line.startswith("Categorías:"):
                data["categories"] = line.replace("Categorías:", "").strip()
        return data
    else:
        return {}

# Modificar modelo_llm  para devolver respuestas formateadas
def modelo_llm(question):
    """Función para consultar directamente al modelo de lenguaje."""
    intention = clasificar_intension(question)  # Detectar intención
    response = llm.invoke(f"Consulta: {question}\nResponde con claridad sobre la intención detectada.")
    if intention in RESPONSE_TEMPLATES:
        return respuesta(intention, {"response": response})
    return response

def clasificar_intension(question):
    """Clasifica la intención del usuario."""
    question = question.lower()
    
    # Palabras clave relacionadas con productos
    product_keywords = ["buscar", "tienes", "producto", "disponible", "tienda", "pagina", "precio", "comprar", "encontrar"]
    category_keywords = ["categorías", "tipos de producto", "clases de producto", "variedad"]
    promotion_keywords = ["promociones", "ofertas", "descuentos"]

    if any(keyword in question for keyword in product_keywords):
        return "search_product"
    elif any(keyword in question for keyword in category_keywords):
        return "list_categories"
    elif any(keyword in question for keyword in promotion_keywords):
        return "check_promotions"
    else:
        return "unknown_intent"


# Nodo para el asistente
def assistant(state: MessagesState):
    """Asistente que interactúa con LLM y herramientas."""
    user_message = state["messages"][-1].content
    intent = clasificar_intension(user_message)
    
    if intent == "search_product":
        response = respuesta_embeddings(user_message)
    elif intent == "list_categories" or intent == "check_promotions":
        response = respuesta_embeddings(user_message)
    else:
        response = modelo_llm(user_message)
    
    return {"messages": state["messages"] + [SystemMessage(content=str(response))]}
    


# Configuración de herramientas
tools = (respuesta_embeddings, modelo_llm)

# Crear el mensaje del sistema
sys_msg = SystemMessage(content="Eres un asistente especializado en productos de belleza.")

# Construcción del gráfico con LangGraph
builder = StateGraph(MessagesState)

# Agregar nodos
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Añadir transiciones y condiciones
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    tools_condition
)
builder.add_edge("tools", "assistant")

# Compilar el gráfico
graph = builder.compile()

#  Streamlit
st.title("Asistente profesional de Cosméticos Rosana")

# Ingreso de pregunta por parte del usuario
question = st.text_input("¿Qué producto deseas buscar o consultar?")

if question:
    # Realizar consulta al modelo
    initial_state = {
        'messages': [HumanMessage(content=question)]
    }
    messages = graph.invoke(initial_state)

    # Mostrar las respuestas
    for m in messages["messages"]:
        st.write(m.content)
      
  
