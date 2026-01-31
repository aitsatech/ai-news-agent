import os
import requests
import random
import datetime
import re
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Define the Shared State
class AgentState(TypedDict):
    field: str
    topic: str
    research: str
    outline: List[str]
    content: str
    iteration: int
    image_url: str

# 2. Setup Models
llm_strategy = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
llm_writer = ChatGroq(model_name="llama3-8b-8192", temperature=0.5)
search = DuckDuckGoSearchRun()

# --- TASK FORCE NODES ---

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
    # FIX: Explicitly told NOT to include "H2" labels in the output
    prompt = (
        f"Create a 4-section outline for '{state['topic']}'. "
        "Return ONLY the titles, one per line. Do NOT include labels like 'H2', 'Section', or numbers."
    )
    # Filter out any lingering "H2" or section labels
    raw_sections = llm_strategy.invoke(prompt).content.split('\n')
    sections = [re.sub(r'^(H2|Section|Step)\s*:?\s*', '', s, flags=re.I).strip() for s in raw_sections if len(s.strip()) > 5]
    return {"outline": sections[:4]}

def writer_node(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer: Drafting {current_section}...")
    # FIX: Instructing writer NOT to repeat the title to avoid duplication
    prompt = f"""Write a section for: {current_section}. 
    Research: {state['research']}. 
    RULES: 
    1. DO NOT repeat the title or include any headers in your response.
    2. Bold the first sentence. 
    3. Max 250 words. 
    4. Use 1 bulleted list."""
    
    res = llm_writer.invoke(prompt).content.strip()
    
    # Manually adding the header here ensures clean formatting without duplication
    section_md = f"\n\n## {current_section}\n\n{res}"
    return {"content": state['content'] + section_md, "iteration": state['iteration'] + 1}

def aio_editor_node(state: AgentState):
    print("📋 AIO Editor: Drafting the 'Key Takeaways' box...")
    prompt = f"Summarize this in 3 bullet points: {state['content'][:1000]}."
    box = llm_strategy.invoke(prompt).content
    return {"content": "> ### Key Takeaways\n>\n" + box + "\n\n" + state['content']}

def designer_node(state: AgentState):
    print("🎨 Designer: Generating custom header image...")
    img_prompt = state['topic'].replace(" ", "-")[:50]
    image_url = f"https://image.pollinations.ai/prompt/{img_prompt}?width=1280&height=720&nologo=true&seed={random.randint(1,999)}"
    image_md = f"![Header Image]({image_url})\n\n"
    return {"content": image_md + state['content'], "image_url": image_url}

# --- GRAPH LOGIC ---

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

# --- THE JEKYLL PUBLISHING BLOCK ---

if __name__ == "__main__":
    FIELD = "Artificial Intelligence and Robotics"
    print(f"🚀 Launching Newsroom for: {FIELD}")
    
    final_state = app.invoke({"field": FIELD, "topic": "", "content": "", "iteration": 0, "research": ""})
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    clean_topic = "".join(c for c in final_state['topic'] if c.isalnum() or c==' ').strip()
    slug = clean_topic.lower().replace(" ", "-")
    filename = f"_posts/{today}-{slug}.md"
    
    os.makedirs("_posts", exist_ok=True)
    
    # FINAL CLEANUP: Ensure no "H2" tags or triple headers exist in the final string
    clean_content = final_state['content'].replace("## H2", "##")
    clean_content = re.sub(r'\n{3,}', '\n\n', clean_content) # Clean up extra newlines
    
    front_matter = f"""---
layout: post
title: "{final_state['topic']}"
date: {today} 12:00:00 +0200
categories: [AI, News]
tags: [ai, robotics, future]
---

"""
    with open(filename, "w") as f:
        f.write(front_matter + clean_content)
    
    print(f"✅ Success: Published {filename}")
