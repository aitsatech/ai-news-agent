import os
import requests
import random
import datetime
import re
import time
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
    prompt = f"Create a 4-section outline for '{state['topic']}'. Return ONLY titles, one per line."
    raw_sections = llm_strategy.invoke(prompt).content.split('\n')
    # Filter out common AI-generated labels from the outline itself
    sections = [re.sub(r'^(H2|Section|Step|Title|Header|[0-9]\.)\s*:?\s*', '', s, flags=re.I).strip() 
                for s in raw_sections if len(s.strip()) > 5]
    return {"outline": sections[:4]}

def writer_node(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer: Drafting {current_section}...")
    prompt = f"Write a section for: {current_section}. Research: {state['research']}. RULES: 1. No title repetition. 2. Bold 1st sentence. 3. Max 200 words. 4. Use 1 list. 5. No redundant H2 tags inside the text."
    res = llm_writer.invoke(prompt).content.strip()
    # Apply Permanent Solution: Spacing before H2
    section_md = f"\n\n## {current_section}\n\n{res}\n"
    return {"content": state['content'] + section_md, "iteration": state['iteration'] + 1}

def aio_editor_node(state: AgentState):
    print("📋 Editor: Finalizing content and takeaways...")
    prompt = f"Summarize this in 3 short bullet points: {state['content'][:1000]}."
    box = llm_strategy.invoke(prompt).content
    # Apply Permanent Solution: Extra padding for layout clarity
    header_box = f"> ### Key Takeaways\n>\n{box}\n\n&nbsp;\n\n" 
    return {"content": header_box + state['content']}

def designer_node(state: AgentState):
    print("🎨 Designer: Generating dynamic header...")
    img_dir = "assets/img"
    os.makedirs(img_dir, exist_ok=True)
    safe_topic = re.sub(r'[^a-zA-Z0-9]', '-', state['topic'][:20]).lower()
    img_filename = f"header-{safe_topic}-{random.randint(100,999)}.png"
    img_path = os.path.join(img_dir, img_filename)
    
    # Retry Logic for 429 Errors + Apex Fallback
    for attempt in range(2):
        try:
            model_to_use = "gemini-2.0-flash" if attempt == 0 else "gemini-1.5-flash"
            img_prompt = f"A wide, high-tech, cinematic 4k header image for: {state['topic']}. Professional, no text."
            response = client.models.generate_content(
                model=model_to_use,
                contents=[img_prompt],
                config=types.GenerateContentConfig(image_config=types.ImageConfig(aspect_ratio="16:9"))
            )
            if response.generated_images:
                with open(img_path, "wb") as f:
                    f.write(response.generated_images[0].image_bytes)
                return {"image_url": img_filename}
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} image gen failed: {e}")
            if attempt == 0: time.sleep(5)
    
    # Apex Fallback Image
    print("🚀 Using Apex Fallback Image to ensure visual integrity.")
    return {"image_url": "https://images.unsplash.com/photo-1677442136019-21780ecad995?auto=format&fit=crop&q=80&w=1000"}

def update_site_branding_avatar(state: AgentState):
    avatar_path = "assets/img/avatar.png"
    if os.path.exists(avatar_path): return
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=["A minimalist high-tech robot avatar logo, white background"],
            config=types.GenerateContentConfig(image_config=types.ImageConfig(aspect_ratio="1:1"))
        )
        with open(avatar_path, "wb") as f:
            f.write(response.generated_images[0].image_bytes)
    except: pass

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
workflow.add_conditional_edges("writer", lambda x: "writer" if x['iteration'] < len(x['outline']) else "editor")
workflow.add_edge("editor", "designer")
workflow.add_edge("designer", END)
app = workflow.compile()

# --- PUBLISHING ---
if __name__ == "__main__":
    final_state = app.invoke({"field": "AI and Robotics", "topic": "", "research": "", "outline": [], "content": "", "iteration": 0, "image_url": ""})
    update_site_branding_avatar(final_state)
    
    # APEX FIX: Force time to early morning to prevent "Future Post" build blocks
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    jekyll_time = f"{today_date} 00:01:00 +0200"
    
    clean_topic = final_state['topic'].replace('"', '')
    slug = re.sub(r'[^a-z0-9]', '-', clean_topic.lower())[:40].strip("-")
    
    # Permanent Solution: Deduplication & Cleaning
    raw_content = final_state['content']
    lines = raw_content.split('\n')
    seen = set()
    deduped_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped == "" or stripped not in seen:
            deduped_lines.append(line)
            if stripped != "" and not stripped.startswith(">") and not stripped.startswith("##"):
                seen.add(stripped)
    
    content = "\n".join(deduped_lines)
    
    # Permanent Solution: Prevent H2 tags next to headers
    content = re.sub(r'##\s*(H2|Header|Section|Title|Topic|Step):?\s*', '## ', content, flags=re.I)
    content = re.sub(r'\n{3,}', '\n\n', content).strip()
    
    # Image Metadata Fix
    image_url = final_state.get('image_url')
    image_meta = ""
    if image_url:
        path_prefix = "/assets/img/" if not image_url.startswith("http") else ""
        image_meta = f"image:\n  path: {path_prefix}{image_url}\n  alt: \"{clean_topic}\""

    post_md = f"""---
layout: post
title: "{clean_topic}"
date: {jekyll_time}
categories: [AI, Technology]
{image_meta}
---

{content}"""

    # Final Save
    os.makedirs("_posts", exist_ok=True)
    filename = f"{today_date}-{slug}.md"
    with open(f"_posts/{filename}", "w", encoding="utf-8") as f:
        f.write(post_md)
        
    print(f"🚀 Article Published: {filename}")
    print(f"⏰ Build Time set to {jekyll_time} to bypass future-date locks.")
