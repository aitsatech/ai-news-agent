import os
import requests
import random
import datetime
import re
import time
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
# Strategy uses a larger model for reasoning, Writer uses a faster one for speed
llm_strategy = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.2)
llm_writer = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.6)

# --- SEARCH RETRY WRAPPER ---
def safe_search(query: str, max_retries=3):
    search_tool = DuckDuckGoSearchRun()
    for i in range(max_retries):
        try:
            return search_tool.run(query)
        except Exception as e:
            print(f"⚠️ Search attempt {i+1} failed: {e}")
            time.sleep(2)
    return "No recent search data available due to connection timeout."

# --- NODES ---

def trend_scout_node(state: AgentState):
    print(f"📡 Trend Scout: Finding viral breakthroughs...")
    news = safe_search(f"latest trending breakthroughs in {state['field']} 2026")
    prompt = f"Based on this news: {news}\nPick the SINGLE most viral story for a deep-dive. Return ONLY the title. No quotes."
    topic = llm_strategy.invoke(prompt).content.strip().replace('"', '')
    return {"topic": topic}

def researcher_node(state: AgentState):
    print(f"🕵️ Researcher: Deep-diving into '{state['topic']}'...")
    data = safe_search(f"{state['topic']} technical details 2026")
    return {"research": data}

def architect_node(state: AgentState):
    print("📐 Architect: Planning high-impact structure...")
    prompt = f"""Create a 4-section outline for an article titled '{state['topic']}'. 
    Ensure sections follow a logical narrative. Return ONLY the titles, one per line. 
    Do not use 'Section 1' or 'Introduction' labels."""
    raw_sections = llm_strategy.invoke(prompt).content.split('\n')
    sections = [re.sub(r'^(H2|Section|Step|Title|Header|[0-9]\.)\s*:?\s*', '', s, flags=re.I).strip() 
                for s in raw_sections if len(s.strip()) > 5]
    return {"outline": sections[:4]}

def writer_node(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer: Drafting {current_section}...")
    
    # REVAMPED PROMPT: Witty, concise, and strict on formatting
    prompt = f"""Role: Witty Tech Journalist. 
    Topic: {current_section} (Part of a larger piece on {state['topic']}).
    Context: {state['research']}
    
    RULES:
    1. Start with a bold, punchy sentence.
    2. Be insightful, avoid "corporate fluff."
    3. Use 1 bulleted list for data/facts.
    4. NO 'H2:' or 'Header:' text.
    5. Do NOT repeat facts mentioned in previous research notes.
    6. Max 150 words.
    """
    
    res = llm_writer.invoke(prompt).content.strip()
    
    # Clean up any AI artifacts immediately
    res = re.sub(r'^(Here is|This section|Sure).*?\n', '', res, flags=re.I)
    res = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', res, flags=re.I)
    
    section_md = f"\n\n## {current_section}\n\n{res}\n"
    return {"content": state['content'] + section_md, "iteration": state['iteration'] + 1}

def aio_editor_node(state: AgentState):
    print("📋 Editor: Scrubbing content and generating insights...")
    prompt = f"""Summarize this content into 3 'Executive Takeaways'. 
    Make them sharp and forward-looking. 
    Content: {state['content']}
    Return ONLY the bullet points."""
    
    box = llm_strategy.invoke(prompt).content.strip()
    box = re.sub(r'^(Here are|Sure|In summary).*?\n', '', box, flags=re.I).strip()
    
    header_box = f"> ### Key Takeaways\n>\n{box}\n\n&nbsp;\n\n" 
    return {"content": header_box + state['content']}

def designer_node(state: AgentState):
    print("🎨 Designer: Generating Banner Asset...")
    img_dir = "assets/img"
    os.makedirs(img_dir, exist_ok=True)
    
    timestamp = int(time.time())
    img_filename = f"header-{timestamp}.png"
    img_path = os.path.join(img_dir, img_filename)
    
    # Try Gemini Image Generation First
    try:
        img_prompt = f"Futuristic, cinematic digital art representing {state['topic']}. 16:9 ratio, ultra-detailed, no text, neon accents."
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[img_prompt],
            config=types.GenerateContentConfig(image_config=types.ImageConfig(aspect_ratio="16:9"))
        )
        if response.generated_images:
            with open(img_path, "wb") as f:
                f.write(response.generated_images[0].image_bytes)
            return {"image_url": img_filename}
    except Exception as e:
        print(f"⚠️ Gemini Image failed, using Unsplash fallback...")

    # Fallback to Unsplash
    try:
        keyword = state['topic'].split()[0]
        fallback_url = f"https://images.unsplash.com/featured/1200x675?{keyword},tech&sig={random.randint(1,999)}"
        res = requests.get(fallback_url, timeout=10)
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

# --- EXECUTION & CLEANUP ---
if __name__ == "__main__":
    final_state = app.invoke({
        "field": "AI and Data Science", 
        "topic": "", "research": "", "outline": [], "content": "", "iteration": 0, "image_url": ""
    })
    
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    
    # Clean topic for title and slug
    clean_topic = final_state['topic'].replace('"', '').strip()
    slug = re.sub(r'[^a-z0-9]', '-', clean_topic.lower()).strip("-")[:50]
    
    # 1. PERMANENT FORMATTING SCRUB
    content = final_state['content']
    # Remove H2 artifacts
    content = re.sub(r'##\s*(H2|Header|Section|Topic):?\s*', '## ', content, flags=re.I)
    # Remove duplicate consecutive words
    content = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', content, flags=re.I)
    
    # 2. DUPLICATE LINE PROTECTION
    lines = content.splitlines()
    final_lines = []
    for line in lines:
        if not final_lines or line.strip() != final_lines[-1].strip() or line.strip() == "":
            final_lines.append(line)
    content = "\n".join(final_lines)

    # 3. CONSTRUCT FRONT MATTER
    image_block = ""
    if final_state['image_url']:
        # Proper Chirpy Image Block
        image_block = f"image:\n  path: /assets/img/{final_state['image_url']}\n  alt: \"{clean_topic}\""

    post_md = f"""---
layout: post
title: "{clean_topic}"
date: {today_date} 09:00:00 +0200
categories: [AI, Technology]
{image_block}
---

{content}"""

    os.makedirs("_posts", exist_ok=True)
    filename = f"{today_date}-{slug}.md"
    with open(f"_posts/{filename}", "w", encoding="utf-8") as f:
        f.write(post_md)
        
    print(f"✅ Success! Article created: {filename}")
