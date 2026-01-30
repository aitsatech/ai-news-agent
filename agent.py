import os
import requests
import random
import datetime
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Define the Shared State
class AgentState(TypedDict):
    field: str        # The broad area of interest
    topic: str        # The specific trending story selected
    research: str     # Collected data
    outline: List[str]# Section titles
    content: str      # The growing markdown body
    iteration: int    # Progress tracker
    image_url: str    # The final image link

# 2. Setup Models & Search
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
search = DuckDuckGoSearchRun()

# --- TASK FORCE NODES ---

def trend_scout_node(state: AgentState):
    print(f"📡 Trend Scout: Finding viral breakthroughs in {state['field']}...")
    news = search.run(f"latest trending breakthroughs and news in {state['field']} 2026")
    prompt = f"Based on this news: {news}\nPick the SINGLE most viral story for a deep-dive. Return ONLY the title."
    topic = llm.invoke(prompt).content.strip().replace('"', '')
    print(f"🎯 Selected Topic: {topic}")
    return {"topic": topic}

def researcher_node(state: AgentState):
    print(f"🕵️ Researcher: Deep-diving into '{state['topic']}'...")
    data = search.run(f"{state['topic']} technical details and expert analysis 2026")
    return {"research": data}

def architect_node(state: AgentState):
    print("📐 Architect: Planning 2,000-word structure...")
    prompt = f"Create a 6-section H2 outline for '{state['topic']}' using research: {state['research']}. Return ONLY titles, one per line."
    sections = [s.strip() for s in llm.invoke(prompt).content.split('\n') if len(s.strip()) > 5]
    return {"outline": sections}

def writer_node(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer: Drafting {current_section}...")
    prompt = f"""Write a section for: {current_section}. 
    RULES: Answer-First (start with bold answer), short paragraphs (max 3 sentences), use bullet points.
    Research: {state['research']}. Length: 400 words."""
    res = llm.invoke(prompt).content
    return {"content": state['content'] + f"\n\n## {current_section}\n" + res, "iteration": state['iteration'] + 1}

def aio_editor_node(state: AgentState):
    print("📋 AIO Editor: Drafting the 'Key Takeaways' box...")
    prompt = f"Create a Markdown blockquote (>) with 3 bullet points summarizing: {state['content'][:1500]}."
    box = llm.invoke(prompt).content
    return {"content": box + "\n\n" + state['content']}

def designer_node(state: AgentState):
    print("🎨 Designer: Generating custom header image...")
    img_prompt = llm.invoke(f"Short 10-word art prompt for: {state['topic']}").content.replace(" ", "-")
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
    
    final_state = app.invoke({"field": FIELD, "topic": "", "content": "", "iteration": 0})
    
    # Format for Jekyll
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    slug = final_state['topic'].lower().replace(" ", "-").replace(":", "")
    filename = f"_posts/{today}-{slug}.md"
    
    os.makedirs("_posts", exist_ok=True)
    
    front_matter = f"""---
layout: post
title: "{final_state['topic']}"
date: {today}
categories: AI News
---

"""
    with open(filename, "w") as f:
        f.write(front_matter + final_state['content'])
    
    print(f"✅ Success: Published {filename}")
