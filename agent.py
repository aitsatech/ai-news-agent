import os
import requests
import random
import datetime
import re
from io import BytesIO
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from google import genai
from google.genai import types

# Initialize Gemini Client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class AgentState(TypedDict):
    field: str
    topic: str
    research: str
    outline: List[str]
    content: str
    iteration: int
    image_url: str

# Models
llm_strategy = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
llm_writer = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)
search = DuckDuckGoSearchRun()

# --- NODES ---

def trend_scout_node(state: AgentState):
    print(f"📡 Trend Scout: Finding viral breakthroughs...")
    news = search.run(f"latest trending breakthroughs in {state['field']} 2026")
    prompt = f"Based on this news: {news}\nPick the SINGLE most viral story for a deep-dive. Return ONLY the title."
    topic = llm_strategy.invoke(prompt).content.strip().replace('"', '')
    print(f"🎯 Selected Topic: {topic}")
    return {"topic": topic}

def researcher_node(state: AgentState):
    print(f"🕵️ Researcher: Deep-diving into '{state['topic']}'...")
    data = search.run(f"{state['topic']} technical details 2026")
    return {"research": data}

def architect_node(state: AgentState):
    print("📐 Architect: Planning high-impact structure...")
    prompt = (
        f"Create a 4-section outline for '{state['topic']}'. "
        "Return ONLY the titles, one per line. No labels like 'H2' or 'Section'."
    )
    raw_sections = llm_strategy.invoke(prompt).content.split('\n')
    sections = [re.sub(r'^(H2|Section|Step|Title|Header|[0-9]\.)\s*:?\s*', '', s, flags=re.I).strip() 
                for s in raw_sections if len(s.strip()) > 5]
    return {"outline": sections[:4]}

def writer_node(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer: Drafting {current_section}...")
    prompt = f"""Write a section for: {current_section}. 
    Research: {state['research']}. 
    RULES: 
    1. DO NOT repeat the title.
    2. Bold the first sentence. 
    3. Max 200 words. 
    4. Use 1 bulleted list."""
    
    res = llm_writer.invoke(prompt).content.strip()
    section_md = f"\n\n## {current_section}\n\n{res}"
    return {"content": state['content'] + section_md, "iteration": state['iteration'] + 1}

def aio_editor_node(state: AgentState):
    print("📋 AIO Editor: Drafting the 'Key Takeaways' box...")
    prompt = f"Summarize this in 3 bullet points: {state['content'][:1000]}."
    box = llm_strategy.invoke(prompt).content
    return {"content": "> ### Key Takeaways\n>\n" + box + "\n\n" + state['content']}

def update_site_branding_avatar(state: AgentState):
    print("🎨 Branding: Updating site logo/avatar...")
    avatar_dir = "assets/img"
    os.makedirs(avatar_dir, exist_ok=True)
    avatar_path = os.path.join(avatar_dir, "avatar.png")
    avatar_prompt = "A minimalist, high-tech circular vector logo for an AI news site. 4k, clean lines, cyberpunk blue and white, no text."

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[avatar_prompt],
            config=types.GenerateContentConfig(image_config=types.ImageConfig(aspect_ratio="1:1"))
        )
        if response.generated_images:
            image = response.generated_images[0]
            with open(avatar_path, "wb") as f:
                f.write(image.image_bytes)
            print("✅ Site avatar updated.")
    except Exception as e:
        print(f"⚠️ Gemini Avatar Failed: {e}")
    return state

def designer_node(state: AgentState):
    print("🎨 Designer: Generating dynamic header...")
    img_dir = "assets/img"
    os.makedirs(img_dir, exist_ok=True)
    
    safe_topic = re.sub(r'[^a-zA-Z0-9]', '-', state['topic'][:20]).lower()
    img_filename = f"header-{safe_topic}-{random.randint(100,999)}.png"
    img_path = os.path.join(img_dir, img_filename)
    
    try:
        img_prompt = f"A wide, high-tech, cinematic 4k header image for an article about: {state['topic']}. No text, professional digital art."
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[img_prompt],
            config=types.GenerateContentConfig(image_config=types.ImageConfig(aspect_ratio="16:9"))
        )
        
        if response.generated_images:
            image = response.generated_images[0]
            with open(img_path, "wb") as f:
                f.write(image.image_bytes)
            print(f"✨ Success: Gemini generated header: {img_filename}")
            return {"content": state['content'], "image_url": img_filename}
                
    except Exception as e:
        print(f"⚠️ Image Generation Failed: {e}")
    return {"content": state['content'], "image_url": ""}

# --- GRAPH ---
workflow = StateGraph(AgentState)
workflow.add_node("scout", trend_scout_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("architect", architect_node)
workflow.add_node("writer", writer_node)
workflow.add_node("editor", aio_editor_node)
workflow.add_node("designer", designer_node)

workflow.set_entry_point("scout")
workflow.add_edge("scout", "researcher")
workflow.add_edge("researcher", "architect")
workflow.add_edge("architect", "writer")

def loop_check(state):
    return "writer" if state['iteration'] < len(state['outline']) else "editor"

workflow.add_conditional_edges("writer", loop_check)
workflow.add_edge("editor", "designer")
workflow.add_edge("designer", END)
app = workflow.compile()

# --- PUBLISHING ---
if __name__ == "__main__":
    FIELD = "Artificial Intelligence and Robotics"
    final_state = app.invoke({
        "field": FIELD, "topic": "", "content": "", "iteration": 0, 
        "research": "", "outline": [], "image_url": ""
    })
    
    update_site_branding_avatar(final_state)
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    slug = re.sub(r'[^a-z0-9]', '-', final_state['topic'].lower())[:40].strip("-")
    filename = f"_posts/{today}-{slug}.md"
    
    # 1. PERMANENT H2 FIX: Clean H2 tag hallucinations
    clean_content = final_state['content']
    clean_content = re.sub(r'##\s*(H2|Header|Section|Title|Topic|Step):?\s*', '## ', clean_content, flags=re.I)
    
    # 2. PERMANENT DUPLICATE LINE FIX: Collapse whitespace
    clean_content = re.sub(r'\n{3,}', '\n\n', clean_content).strip()
    
    # --- CHIRPY BANNER FRONT MATTER FIX ---
    # This is the part that was missing in your last run
    image_meta = ""
    if final_state.get('image_url'):
        # We ensure the path is absolute from the site root for Chirpy
        image_meta = f"image:\n  path: /assets/img/{final_state['image_url']}\n  alt: \"{final_state['topic']}\""

    fm = f"""---
layout: post
title: "{final_state['topic']}"
date: {today} 12:00:00 +0200
categories: [AI]
{image_meta}
---

"""
    # Create directory if it doesn't exist
    os.makedirs("_posts", exist_ok=True)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(fm + clean_content)
    
    print(f"✅ Success: Published {filename} with banner metadata.")
