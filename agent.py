import os
import requests
import random
import datetime
import re
import time
import urllib.parse
from typing import TypedDict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# --- CONFIGURATION: THE SOVEREIGN SYNDICATE ---
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
    quality_checks: int
    image_url: str
    seo_keywords: str
    content: str  # FIXED: Added missing key to prevent KeyError

# --- SEARCH WRAPPER ---
def safe_search(query: str, max_retries=3):
    search_tool = DuckDuckGoSearchRun()
    for i in range(max_retries):
        try:
            return search_tool.run(query)
        except Exception as e:
            print(f"⚠️ Search attempt {i+1} failed: {e}")
            time.sleep(2)
    return "Data unavailable. Proceed with analytical inference."

# --- NODE 1: RESEARCHER DELUXE ---
def deep_data_diviner(state: AgentState):
    print(f"🕵️ Researcher Deluxe: Initiating deep-scan for '{state['field']}'...")
    query = f"latest emerging technological breakthroughs {state['field']} 2026 technical analysis"
    raw_data = safe_search(query)
    
    prompt = f"Analyze: {raw_data}\nIdentify the ONE most provocative, avant-garde topic. Return ONLY the topic title."
    topic = llm_sovereign.invoke(prompt).content.strip().replace('"', '')
    print(f"   ↳ Topic Identified: {topic}")
    
    technical_data = safe_search(f"{topic} technical specifications implications 2026")
    return {"topic": topic, "research_data": technical_data}

# --- NODE 2: SEO APEX STRATEGIST ---
def seo_apex_strategist(state: AgentState):
    print("🌐 SEO Sovereign: Mapping neural search entities...")
    prompt = f"Generate 5 semantic SEO keywords for: '{state['topic']}'. Comma-separated list ONLY."
    keywords = llm_alchemist.invoke(prompt).content.strip()
    return {"seo_keywords": keywords}

# --- NODE 3: MASTER EDITOR-IN-CHIEF ---
def master_editor(state: AgentState):
    print("🏛️ Master Editor: Calibrating the 'Soul' of the narrative...")
    prompt = f"""
    Create a 4-section outline for: {state['topic']}
    TONE: Ruthlessly Intellectual, Avant-Garde.
    Return ONLY 4 section titles, one per line. No numbering, no prefixes.
    """
    raw_out = llm_sovereign.invoke(prompt).content
    sections = [line.strip() for line in raw_out.split('\n') if len(line.strip()) > 5]
    return {"outline": sections[:4], "iteration": 0, "full_draft": "", "content": ""}

# --- NODE 4: PROMPT ENGINEER COMMANDER ---
def prompt_commander(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"👻 Ghost in the Machine: Designing Prime Directives for '{current_section}'...")
    
    prompt = f"""
    ROLE: Apex Tech Journalist.
    SECTION: {current_section}
    CONTEXT: {state['research_data']}
    KEYWORDS: {state['seo_keywords']}
    
    PRIME DIRECTIVES:
    1. DO NOT include the section title or any headers in your response.
    2. Start immediately with the analysis.
    3. TONE: Ruthlessly Intellectual. 
    4. Max 180 words.
    """
    draft = llm_alchemist.invoke(prompt).content.strip()
    return {"section_content": draft, "current_section": current_section}

# --- NODE 5: PROOFREADER GENIUS ---
def syntax_sentinel(state: AgentState):
    print(f"🧐 Syntax Sentinel: Auditing section {state['iteration'] + 1}...")
    content = state['section_content']
    
    # Audit for fluff
    verdict_prompt = f"Does this text contain corporate fluff or clichés? Text: {content}\nReturn ONLY 'APPROVED' or 'REJECTED'."
    verdict = llm_sovereign.invoke(verdict_prompt).content.strip().upper()
    
    if "REJECTED" in verdict and state.get('quality_checks', 0) < 2:
        print("   ❌ REJECTED. Demanding rewrite...")
        return {"quality_checks": state.get('quality_checks', 0) + 1}
    
    # PERMANENT SOLUTION: Strip LLM-generated headers and titles to prevent duplicates
    clean_content = re.sub(r'^(#+.*|' + re.escape(state['current_section']) + r'.*)\n', '', content, flags=re.I | re.M).strip()
    
    print("   ✅ APPROVED. Integrating to Matrix.")
    formatted_section = f"\n\n## {state['current_section']}\n\n{clean_content}\n"
    
    return {
        "full_draft": state['full_draft'] + formatted_section, 
        "iteration": state['iteration'] + 1,
        "quality_checks": 0
    }

# --- NODE 6: PUBLISHING KING ---
def publishing_king(state: AgentState):
    print("👑 Publishing King: Finalizing Omni-Channel Distribution...")
    
    # Generate Visual
    timestamp = int(time.time())
    img_filename = f"apex-{timestamp}.png"
    img_dir = "assets/img"
    os.makedirs(img_dir, exist_ok=True)
    
    vis_prompt = f"Abstract avant-garde cinematic image for '{state['topic']}'. 10 words."
    visual_concept = llm_alchemist.invoke(vis_prompt).content.strip().replace('"', '')
    encoded_prompt = urllib.parse.quote(f"{visual_concept}, 8k, futuristic")
    gen_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&seed={timestamp}&nologo=true"
    
    try:
        img_data = requests.get(gen_url, timeout=20).content
        with open(os.path.join(img_dir, img_filename), "wb") as f:
            f.write(img_data)
        final_img_url = img_filename
    except:
        final_img_url = ""

    # Final Polish
    summary_prompt = f"Summarize this into 3 Prophetic Insights: {state['full_draft']}"
    takeaways = llm_sovereign.invoke(summary_prompt).content.strip()
    takeaways = re.sub(r'^(Here are|Sure).*?\n', '', takeaways, flags=re.I)
    
    final_content = f"> ### Prophetic Insights\n>\n{takeaways}\n\n{state['full_draft']}"
    
    # PERMANENT SOLUTION: Clean H2 tags and redundant prefixes
    final_content = re.sub(r'##\s*(H2|Header|Section):?\s*', '## ', final_content, flags=re.I)
    
    return {"content": final_content, "image_url": final_img_url}

# --- GRAPH ---
def quality_router(state: AgentState):
    if state.get('quality_checks', 0) > 0: return "rewrite"
    if state['iteration'] < len(state['outline']): return "next"
    return "finalize"

workflow = StateGraph(AgentState)
workflow.add_node("researcher", deep_data_diviner)
workflow.add_node("seo_strategist", seo_apex_strategist)
workflow.add_node("editor", master_editor)
workflow.add_node("writer", prompt_commander)
workflow.add_node("auditor", syntax_sentinel)
workflow.add_node("publisher", publishing_king)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "seo_strategist")
workflow.add_edge("seo_strategist", "editor")
workflow.add_edge("editor", "writer")
workflow.add_edge("writer", "auditor")
workflow.add_conditional_edges("auditor", quality_router, {"rewrite": "writer", "next": "writer", "finalize": "publisher"})
workflow.add_edge("publisher", END)
app = workflow.compile()

# --- EXECUTION ---
if __name__ == "__main__":
    print("🔮 THE SOVEREIGN INTELLIGENCE SYNDICATE IS ONLINE.")
    initial_state = {
        "field": "Artificial Intelligence & Robotics", "topic": "", "research_data": "",
        "seo_keywords": "", "outline": [], "current_section": "", "section_content": "",
        "full_draft": "", "iteration": 0, "quality_checks": 0, "image_url": "", "content": ""
    }
    
    final_state = app.invoke(initial_state)
    
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    slug = re.sub(r'[^a-z0-9]', '-', final_state['topic'].lower()).strip("-")[:50]
    
    img_path_str = f"/assets/img/{final_state['image_url']}" if final_state['image_url'] else ""
    
    post_md = f"""---
layout: post
title: "{final_state['topic']}"
date: {today_date} 09:00:00 +0200
categories: [Apex, Technology]
image:
  path: {img_path_str}
  alt: "{final_state['topic']}"
---

{final_state['content']}"""

    os.makedirs("_posts", exist_ok=True)
    with open(f"_posts/{today_date}-{slug}.md", "w", encoding="utf-8") as f:
        f.write(post_md)
    print(f"🚀 PUBLISHED: {today_date}-{slug}.md")
