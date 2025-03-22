import streamlit as st
import sqlite3
import requests

# === Get zone data from SQLite ===
def get_zone_data(zone_name):
    conn = sqlite3.connect("wildfire_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT ndvi, nbr, ndwi, summary FROM zone_risk WHERE zone_name = ?", (zone_name,))
    result = cursor.fetchone()
    conn.close()
    return result

# === Ask Mistral via Ollama ===
def ask_mistral(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma:2b", "prompt": prompt}
    )
    return response.json()['response']

# === Combine everything into a chatbot flow ===
def wildfire_chatbot(zone_name, user_question):
    data = get_zone_data(zone_name)
    if not data:
        return f"No data found for zone '{zone_name}'."
    
    ndvi, nbr, ndwi, summary = data

    prompt = f"""
You are a wildfire risk assistant operating offline.

Zone: {zone_name}
NDVI: {ndvi}
NBR: {nbr}
NDWI: {ndwi}
Summary: {summary}

User Question: {user_question}

Answer in clear and helpful language:
"""
    return ask_mistral(prompt)

# === Streamlit UI ===
st.set_page_config(page_title="Wildfire Risk Chatbot", layout="centered")
st.title("🔥 Offline Wildfire Risk Chatbot")

with st.form("chat_form"):
    zone_input = st.text_input("Enter Zone Name (e.g., Zone A)")
    question_input = st.text_area("Ask a question about this zone", height=120)
    submitted = st.form_submit_button("Ask")

if submitted:
    if zone_input and question_input:
        with st.spinner("Thinking..."):
            reply = wildfire_chatbot(zone_input.strip(), question_input.strip())
        st.markdown("### 💬 Chatbot Response")
        st.write(reply)
    else:
        st.warning("Please enter both a zone name and a question.")
