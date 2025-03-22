import sqlite3
import requests

# Step 1: Fetch data from SQLite
def get_zone_data(zone_name):
    conn = sqlite3.connect("wildfire_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT ndvi, nbr, ndwi, summary FROM zone_risk WHERE zone_name = ?", (zone_name,))
    result = cursor.fetchone()
    conn.close()
    return result

# Step 2: Query Ollama (Mistral)
def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma:2b", "prompt": prompt}
    )
    return response.json()['response']

# Step 3: Combine into chatbot flow
def wildfire_chatbot(zone, question):
    data = get_zone_data(zone)
    if not data:
        return f"No data found for zone '{zone}'."
    
    ndvi, nbr, ndwi, summary = data
    prompt = f"""
You are an offline wildfire risk assistant.

Zone: {zone}
NDVI: {ndvi}
NBR: {nbr}
NDWI: {ndwi}
Summary: {summary}

User question: {question}
Answer:"""
    return ask_llm(prompt)

# Test it
if __name__ == "__main__":
    zone = input("Enter zone (e.g. Zone A): ")
    question = input("Ask your question: ")
    response = wildfire_chatbot(zone, question)
    print("\n💬 Chatbot says:\n")
    print(response)
