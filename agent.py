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
    sections = [re.sub(r'^(H2|Section|Step|Title|Header|[0-9]\.)\s*:?\s*', '', s, flags=re.I).strip() 
                for s in raw_sections if len(s.strip()) > 5]
    return {"outline": sections[:4]}

def writer_node(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer: Drafting {current_section}...")
    prompt = f"Write a section for: {current_section}. Research: {state['research']}. RULES: 1. No title repetition. 2. Bold 1st sentence. 3. Max 200 words. 4. Use 1 list. 5. No redundant H2 tags inside the text."
    res = llm_writer.invoke(prompt).content.strip()
    section_md = f"\n\n## {current_section}\n\n{res}\n"
    return {"content": state['content'] + section_md, "iteration": state['iteration'] + 1}

def aio_editor_node(state: AgentState):
    print("📋 Editor: Finalizing content and takeaways...")
    prompt = f"Summarize this in 3 short bullet points: {state['content'][:1000]}."
    box = llm_strategy.invoke(prompt).content
    header_box = f"> ### Key Takeaways\n>\n{box}\n\n&nbsp;\n\n" 
    return {"content": header_box + state['content']}

def designer_node(state: AgentState):
    """REVAMPED: Guarantees a unique image every time via Gemini or Dynamic Fallback."""
    print("🎨 Designer: Generating dynamic header...")
    img_dir = "assets/img"
    os.makedirs(img_dir, exist_ok=True)
    
    # Generate unique filename
    timestamp = int(time.time())
    safe_topic = re.sub(r'[^a-zA-Z0-9]', '-', state['topic'][:15]).lower()
    img_filename = f"header-{safe_topic}-{timestamp}.png"
    img_path = os.path.join(img_dir, img_filename)
    
    # Step 1: Attempt Gemini Gen
    for model_name in ["gemini-2.0-flash", "gemini-1.5-flash"]:
        try:
            img_prompt = f"Cinematic 4k high-tech professional photography of {state['topic']}. Futuristic, no text, wide shot."
            response = client.models.generate_content(
                model=model_name,
                contents=[img_prompt],
                config=types.GenerateContentConfig(image_config=types.ImageConfig(aspect_ratio="16:9"))
            )
            if response.generated_images:
                with open(img_path, "wb") as f:
                    f.write(response.generated_images[0].image_bytes)
                print(f"✅ Gemini Image Generated: {img_filename}")
                return {"image_url": img_filename}
        except Exception as e:
            print(f"⚠️ {model_name} failed: {e}")
            continue

    # Step 2: Dynamic Fallback (No more static IDs)
    print("🚀 Using Dynamic Fallback...")
    try:
        # We use a random seed + topic keywords to force Unsplash to provide a new image
        keywords = f"technology,{state['field'].replace(' ', ',')},future"
        random_seed = random.randint(1, 10000)
        # Using the Unsplash Source-like redirect URL with a random seed
        fallback_url = f"https://images.unsplash.com/featured/1200x675?{keywords}&sig={random_seed}"
        
        fallback_res = requests.get(fallback_url, timeout=15)
        if fallback_res.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(fallback_res.content)
            print(f"✅ Dynamic Fallback Saved: {img_filename}")
            return {"image_url": img_filename}
    except Exception as e:
        print(f"❌ Fallback failed: {e}")
    
    return {"image_url": ""}

def update_site_branding_avatar(state: AgentState):
    avatar_path = "assets/img/avatar.png"
    if os.path.exists(avatar_path): return
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=["A minimalist high-tech robot avatar logo"],
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
    
    # Time Fix: Ensure date is slightly in the past to prevent Jekyll hiding "Future" posts
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    jekyll_time = f"{today_date} 00:01:00 +0200"
    
    clean_topic = final_state['topic'].replace('"', '')
    # Cleaner Slug logic for better URL compatibility
    slug = re.sub(r'[^a-z0-9]', '-', clean_topic.lower())
    slug = re.sub(r'-+', '-', slug).strip("-")[:50]
    
    # CONTENT CLEANING (Permanent Solution)
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
    content = re.sub(r'##\s*(H2|Header|Section|Title|Topic|Step):?\s*', '## ', content, flags=re.I)
    content = re.sub(r'\n{3,}', '\n\n', content).strip()
    
    # IMAGE METADATA
    image_url = final_state.get('image_url')
    image_meta = ""
    if image_url:
        image_meta = f"image:\n  path: /assets/img/{image_url}\n  alt: \"{clean_topic}\""

    post_md = f"""---
layout: post
title: "{clean_topic}"
date: {jekyll_time}
categories: [AI, Technology]
{image_meta}
---

{content}"""

    os.makedirs("_posts", exist_ok=True)
    filename = f"{today_date}-{slug}.md"
    with open(f"_posts/{filename}", "w", encoding="utf-8") as f:
        f.write(post_md)
        
    print(f"🚀 Published: {filename}")
    print(f"📸 Image assigned: {image_url}")
