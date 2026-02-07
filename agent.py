import os
import requests
import re
import time
import datetime  # Added for timestamping files
import urllib.parse
from google import genai 
from google.genai import types 
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# --- CONFIGURATION ---
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

llm_sovereign = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
llm_alchemist = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)

class AgentState(TypedDict):
    field: str
    topic: str
    research_data: str
    outline: List[str]
    current_section: str
    section_content: str
    full_draft: str
    iteration: int
    image_url: str
    seo_keywords: str
    content: str 

# --- NODES ---

def deep_data_diviner(state: AgentState):
    print(f"🕵️ Researcher: Scouting technical niche...")
    query = f"latest {state['field']} breakthroughs 2026"
    raw_data = DuckDuckGoSearchRun().run(query)
    prompt = f"Data: {raw_data}\nIdentify the single most technical niche topic. Return ONLY the title."
    topic = llm_sovereign.invoke(prompt).content.strip().replace('"', '')
    return {"topic": topic, "research_data": raw_data}

def seo_apex_strategist(state: AgentState):
    prompt = f"Provide 5 SEO keywords for '{state['topic']}'. Comma-separated ONLY."
    keywords = llm_alchemist.invoke(prompt).content.strip()
    return {"seo_keywords": keywords}

def master_editor(state: AgentState):
    print("🏛️ Master Editor: Structuring narrative...")
    prompt = f"Create a 4-section technical outline for: {state['topic']}. Return ONLY section titles, 1 per line."
    raw_out = llm_sovereign.invoke(prompt).content
    sections = [line.strip() for line in raw_out.split('\n') if len(line.strip()) > 5]
    return {"outline": sections[:4], "iteration": 0, "full_draft": ""}

def prompt_commander(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"👻 Writer: Drafting {current_section}...")
    prompt = f"""
    SECTION: {current_section}
    CONTEXT: {state['research_data']}
    CONSTRAINTS: 
    - No headers or titles. 
    - Start immediately with the content.
    - No introductory phrases.
    """
    draft = llm_alchemist.invoke(prompt).content.strip()
    return {"section_content": draft, "current_section": current_section}

def syntax_sentinel(state: AgentState):
    """
    PERMANENT SOLUTION: Hard-strips H2 tags and Duplicate Lines.
    """
    content = state['section_content']
    section_title = state['current_section'].strip()
    
    # 1. Regex Strip
    clean = re.sub(r'(?i)^#+.*$', '', content, flags=re.MULTILINE)
    clean = re.sub(r'(?i)^h2:.*$', '', clean, flags=re.MULTILINE)
    
    # 2. Logic Strip
    lines = clean.split('\n')
    filtered_lines = [l for l in lines if l.strip().lower() != section_title.lower() and l.strip()]
    
    final_body = "\n\n".join(filtered_lines)
    
    formatted_section = f"\n\n## {section_title}\n\n{final_body}\n"
    
    return {
        "full_draft": state['full_draft'] + formatted_section, 
        "iteration": state['iteration'] + 1
    }

def publishing_king(state: AgentState):
    print("👑 Publisher: Finalizing and Saving...")
    
    # --- 1. Image Generation Logic (Maintained) ---
    timestamp = int(time.time())
    img_filename = f"apex-{timestamp}.png"
    img_path = os.path.join("assets/img", img_filename)
    os.makedirs("assets/img", exist_ok=True)
    
    img_prompt = f"Cinematic tech visual for {state['topic']}, 8k, futuristic."
    short_prompt = f"Technology {state['topic']} futuristic"
    success = False
    
    # Provider 1: Gemini
    print("   🎨 Attempting Nano Banana (Imagen 3)...")
    try:
        response = client.models.generate_images(
            model='
