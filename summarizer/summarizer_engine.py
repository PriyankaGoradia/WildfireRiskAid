import requests

ollama_url = "http://localhost:11434/api/generate"
model = "mistral"

def summarize_zone(zone_id, stats):
    ndvi = stats.get("NDVI", {}).get("mean", "unknown") #format not yet specified. will prob change.
    nbr = stats.get("NBR", {}).get("mean", "unknown")
    # name = stats.get
    summary_prompt = f"""
Zone: {zone_id}
NDVI (mean): {ndvi}
NBR (mean): {nbr}

Please generate a concise summary of wildfire risk for this zone. Please mention vegetation density (NDVI), 
burn severity (NBR), and any relevant implications for emergency responders.
"""

    response = requests.post(ollama_url, json={
        "model": model,
        "prompt": summary_prompt,
        "stream": False
    })

    result = response.json()
    return result.get("response", "[No response generated]")
