import re

def extract_knowledge_entities(text):
    # Simplified logic to find Parties in a DPA
    parties = re.findall(r'\((.*?)\)\s+(.*?)(?=,|$)', text[:2000])
    entities = [{"role": p[0], "entity": p[1]} for p in parties]
    return entities 