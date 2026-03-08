import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# By CEEL
# --- 1. CONFIGURACIÓN E INYECCIÓN DE CSS (TU MOCKUP) ---
st.set_page_config(page_title="BIT_Prototipo", layout="centered")

# Control de Modo Oscuro/Claro manual
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True  # Por defecto oscuro como tu imagen

# Colores extraídos de tus imágenes de Figma
bg = "#000000" if st.session_state.dark_mode else "#FFFFFF"
card_bg = "#1A1A1A" if st.session_state.dark_mode else "#F0F2F5"
text = "#FFFFFF" if st.session_state.dark_mode else "#000000"
bubble = "#262626" if st.session_state.dark_mode else "#E9E9EB"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {bg}; color: {text}; }}
    
    /* Header estilo Figma */
    .header-container {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem;
        background-color: {card_bg};
        border-radius: 0 0 20px 20px;
        margin-bottom: 20px;
    }}

    /* Burbujas de Chat Redondeadas */
    .chat-bubble {{
        background-color: {bubble};
        color: {text};
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        margin-bottom: 10px;
        max-width: 85%;
        font-family: 'Segoe UI', sans-serif;
    }}

    /* Botones de Sugerencias (Píldoras) */
    div.stButton > button {{
        background-color: transparent !important;
        color: #1E88E5 !important;
        border: 1px solid #333 !important;
        border-radius: 25px !important;
        padding: 10px 20px !important;
        transition: 0.3s;
        width: 100%;
    }}
    div.stButton > button:hover {{
        border-color: #1E88E5 !important;
        background-color: rgba(30, 136, 229, 0.1) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- 2. HEADER PERSONALIZADO (MOCKUP) ---
with st.container():
    col_back, col_logo, col_toggle = st.columns([1, 4, 1])
    with col_back:
        st.write("### ⬅️")
    with col_logo:
        st.markdown(
            f"<div style='text-align: center;'><b style='font-size: 20px;'>BIT_Prototipo</b><br><small style='color: #4CAF50;'>● En línea</small></div>",
            unsafe_allow_html=True,
        )
    with col_toggle:
        icon = "☀️" if st.session_state.dark_mode else "🌙"
        if st.button(icon):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()


# --- 3. CARGA DEL MOTOR BIT ---
@st.cache_resource
def load_assets():
    params = json.load(open("Parameters.json", "r", encoding="utf-8"))
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    model = load_model("Bit_Chatbot_RRH.keras")  # <- PONER EL NOMBRE DEL BOT
    return params, words, classes, model


params, words, classes, model = load_assets()
lemmatizer = WordNetLemmatizer()

# --- 4. HISTORIAL DE CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "bot",
            "content": "¡Hola! 👋 Soy BIT, tu asistente de RRHH del ITSE. ¿Cómo puedo ayudarte hoy?",
        }
    ]

# Renderizado de mensajes con el estilo Figma
for m in st.session_state.messages:
    if m["role"] == "bot":
        st.markdown(
            f"<div class='chat-bubble'>{m['content']}</div>", unsafe_allow_html=True
        )
    else:
        # Los mensajes del usuario usan el formato nativo para contraste
        st.chat_message("user").write(m["content"])

# --- 5. SECCIÓN DE SUGERENCIAS (Tus botones de Figma) ---
st.write("---")
col1, col2 = st.columns(2)
opciones = [
    ("📅 Vacaciones", "vacaciones"),
    ("🔔 Permisos", "permisos"),
    ("⏰ Horas extras", "horas_Extras"),
    ("📋 Normativas", "normativas"),
]


def click_sugerencia(texto, tag):
    st.session_state.messages.append({"role": "user", "content": texto})
    for intent in params["Parameters"]:
        if intent["tag"] == tag:
            res = random.choice(intent["response"])
            st.session_state.messages.append({"role": "bot", "content": res})
            break
    st.rerun()


for idx, (label, tag) in enumerate(opciones):
    target_col = col1 if idx < 2 else col2
    if target_col.button(label):
        click_sugerencia(label, tag)

# --- 6. INPUT DE TEXTO ---
if prompt := st.chat_input("Escribe tu consulta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Predicción
    tokens = [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(prompt)]
    bag = [1 if w in tokens else 0 for w in words]
    pred = model.predict(np.array([bag]), verbose=0)[0]
    tag = classes[np.argmax(pred)]

    # Respuesta
    res = "Lo siento, no tengo esa información. Intenta con las sugerencias."
    for i in params["Parameters"]:
        if i["tag"] == tag:
            res = random.choice(i["response"])

    st.session_state.messages.append({"role": "bot", "content": res})
    st.rerun()
