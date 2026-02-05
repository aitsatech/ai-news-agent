import os
import requests
import random
import datetime
import re
import time
import urllib.parse
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

class AgentState(TypedDict):
    field: str
    topic: str
    research: str
    outline: List[str]
    content: str
    iteration: int
    image_url: str

# Models
llm_strategy = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
llm_writer = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.6)

def safe_search(query: str, max_retries=3):
    search_tool = DuckDuckGoSearchRun()
    for i in range(max_retries):
        try:
            return search_tool.run(query)
        except Exception as e:
            print(f"⚠️ Search attempt {i+1} failed: {e}")
            time.sleep(2)
    return "No recent search data available."

# --- NODES (SCOUT, RESEARCHER, ARCHITECT, WRITER, EDITOR REMAIN STABLE) ---

def trend_scout_node(state: AgentState):
    print(f"📡 Trend Scout: Finding viral breakthroughs...")
    news = safe_search(f"latest trending breakthroughs in {state['field']} 2026")
    prompt = f"Based on this news: {news}\nPick the SINGLE most viral story. Return ONLY the title."
    topic = llm_strategy.invoke(prompt).content.strip().replace('"', '')
    return {"topic": topic}

def researcher_node(state: AgentState):
    print(f"🕵️ Researcher: Deep-diving into '{state['topic']}'...")
    data = safe_search(f"{state['topic']} technical details 2026")
    return {"research": data}

def architect_node(state: AgentState):
    print("📐 Architect: Planning structure...")
    prompt = f"Create a 4-section outline for '{state['topic']}'. Titles only, one per line."
    raw_sections = llm_strategy.invoke(prompt).content.split('\n')
    sections = [re.sub(r'^(H2|Section|[0-9]\.)\s*:?\s*', '', s, flags=re.I).strip() 
                for s in raw_sections if len(s.strip()) > 5]
    return {"outline": sections[:4]}

def writer_node(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer: Drafting {current_section}...")
    prompt = f"""Role: Witty Tech Journalist. Topic: {current_section}. Context: {state['research']}.
    RULES: Bold opener, 1 bullet list, no 'H2' labels, max 150 words. Be unique."""
    res = llm_writer.invoke(prompt).content.strip()
    res = re.sub(r'^(Here is|This section|Sure).*?\n', '', res, flags=re.I)
    section_md = f"\n\n## {current_section}\n\n{res}\n"
    return {"content": state['content'] + section_md, "iteration": state['iteration'] + 1}

def aio_editor_node(state: AgentState):
    print("📋 Editor: Finalizing content...")
    prompt = f"Summarize in 3 sharp 'Key Takeaways'. Content: {state['content']}"
    box = llm_strategy.invoke(prompt).content.strip()
    header_box = f"> ### Key Takeaways\n>\n{box}\n\n&nbsp;\n\n" 
    return {"content": header_box + state['content']}

def designer_node(state: AgentState):
    """The Apex Zero-Dollar Designer: Real AI Generation via Pollinations"""
    print("🎨 Designer: Executing Apex Creative Protocol...")
    img_dir = "assets/img"
    os.makedirs(img_dir, exist_ok=True)
    
    timestamp = int(time.time())
    img_filename = f"header-{timestamp}.png"
    img_path = os.path.join(img_dir, img_filename)

    # 1. THE PROMPT STYLIST: Create a specific visual description
    print("🎭 Stylist: Designing AI Image Prompt...")
    style_prompt = f"Describe a cinematic, futuristic visual scene for '{state['topic']}'. Focus on lighting and texture. No text. 25 words max."
    visual_description = llm_strategy.invoke(style_prompt).content.strip().replace('"', '')
    
    # 2. THE GENERATOR: Pollinations.ai (Stable Diffusion)
    # This creates a unique image every time using the timestamp as a seed
    encoded_prompt = urllib.parse.quote(f"{visual_description}, high-tech, cinematic lighting, 8k resolution, wide angle")
    gen_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&seed={timestamp}&nologo=true"

    try:
        print(f"🚀 Generating AI Image for: {state['topic']}...")
        response = requests.get(gen_url, timeout=30)
        if response.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(response.content)
            print(f"✅ AI Image Generated and Saved: {img_filename}")
            return {"image_url": img_filename}
    except Exception as e:
        print(f"❌ Generation failed: {e}")

    # 3. SECONDARY FALLBACK: Random Unsplash (Only if Internet is down)
    print("🔄 Fallback: Triggering Randomized Unsplash...")
    try:
        keyword = state['topic'].split()[0]
        fallback_url = f"https://images.unsplash.com/featured/1200x675?{keyword},tech&sig={timestamp}"
        res = requests.get(fallback_url, timeout=15)
        if res.status_code == 200:
            with open(img_path, "wb") as f:
                f.write(res.content)
            return {"image_url": img_filename}
    except:
        pass
        
    return {"image_url": ""}

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

# --- EXECUTION ---
if __name__ == "__main__":
    final_state = app.invoke({
        "field": "AI and Data Science", 
        "topic": "", "research": "", "outline": [], "content": "", "iteration": 0, "image_url": ""
    })
    
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    clean_topic = final_state['topic']
    slug = re.sub(r'[^a-z0-9]', '-', clean_topic.lower()).strip("-")[:50]
    
    # Final Content Scrubbing (Addressing your 2026nd request for no duplicate lines/tags)
    content = final_state['content']
    content = re.sub(r'##\s*(H2|Header|Section|Title):?\s*', '## ', content, flags=re.I)
    
    lines = content.splitlines()
    unique_lines = []
    for line in lines:
        if not unique_lines or line.strip() != unique_lines[-1].strip() or line.strip() == "":
            unique_lines.append(line)
    content = "\n".join(unique_lines)

    # Path logic for Chirpy
    img_display_path = f"/assets/img/{final_state['image_url']}" if final_state['image_url'] else "https://images.unsplash.com/photo-1614728263952-84ea256f9679?w=1200"

    post_md = f"""---
layout: post
title: "{clean_topic}"
date: {today_date} 09:00:00 +0200
categories: [AI, Technology]
image:
  path: {img_display_path}
  alt: "{clean_topic}"
---

{content}"""

    os.makedirs("_posts", exist_ok=True)
    with open(f"_posts/{today_date}-{slug}.md", "w", encoding="utf-8") as f:
        f.write(post_md)
    print(f"✅ Apex Article Published: {today_date}-{slug}.md")
