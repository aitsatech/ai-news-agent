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
# We use the 70b model for high-level reasoning (The Sovereigns)
# We use the 8b model for rapid execution (The Alchemists)
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
    quality_checks: int  # Counter to prevent infinite loops
    image_url: str
    seo_keywords: str

# --- SEARCH WRAPPER (The Deep-Data Diviner Tool) ---
def safe_search(query: str, max_retries=3):
    search_tool = DuckDuckGoSearchRun()
    for i in range(max_retries):
        try:
            return search_tool.run(query)
        except Exception as e:
            print(f"⚠️ Search attempt {i+1} failed: {e}")
            time.sleep(2)
    return "Data unavailable. Proceed with analytical inference."

# --- NODE 1: RESEARCHER DELUXE (The Deep-Data Diviner) ---
def deep_data_diviner(state: AgentState):
    print(f"🕵️ Researcher Deluxe: Initiating deep-scan for '{state['field']}'...")
    
    # Advanced query to find "story within the noise"
    query = f"latest emerging technological breakthroughs {state['field']} 2026 technical analysis"
    raw_data = safe_search(query)
    
    # Synthesize the "Story within the noise"
    prompt = f"""
    Analyze this raw data: {raw_data}
    Identify the ONE most disruptive, avant-garde topic that mainstream media is missing.
    Return ONLY the topic title. Make it intellectual and provocative.
    """
    topic = llm_sovereign.invoke(prompt).content.strip().replace('"', '')
    
    print(f"   ↳ Topic Identified: {topic}")
    
    # Deep dive for technical specifics
    technical_data = safe_search(f"{topic} technical specifications implications 2026")
    
    return {"topic": topic, "research_data": technical_data}

# --- NODE 2: SEO APEX STRATEGIST (The Search Sovereign) ---
def seo_apex_strategist(state: AgentState):
    print("🌐 SEO Sovereign: Mapping neural search entities...")
    prompt = f"""
    Generate 5 semantic SEO entities/keywords for the topic: '{state['topic']}'.
    These must be high-value terms for 2026 search patterns.
    Return a comma-separated list ONLY.
    """
    keywords = llm_alchemist.invoke(prompt).content.strip()
    return {"seo_keywords": keywords}

# --- NODE 3: MASTER EDITOR-IN-CHIEF (The Neural Architect) ---
def master_editor(state: AgentState):
    print("🏛️ Master Editor: Calibrating the 'Soul' of the narrative...")
    
    prompt = f"""
    Act as the Neural Architect. 
    Topic: {state['topic']}
    Data: {state['research_data']}
    
    Create a 4-section outline. 
    TONE: Ruthlessly Intellectual, Avant-Garde, Prophetic.
    The structure must be logical but defy convention.
    Return ONLY the 4 section titles, one per line. No numbering.
    """
    raw_out = llm_sovereign.invoke(prompt).content
    sections = [line.strip() for line in raw_out.split('\n') if len(line.strip()) > 5]
    return {"outline": sections[:4], "iteration": 0, "full_draft": ""}

# --- NODE 4: PROMPT ENGINEER COMMANDER (Ghost in the Machine) ---
def prompt_commander(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"👻 Ghost in the Machine: Designing Prime Directives for '{current_section}'...")
    
    prompt = f"""
    ROLE: Apex Tech Journalist.
    TONE: Ruthlessly Intellectual & Avant-Garde.
    SECTION: {current_section}
    CONTEXT: {state['research_data']}
    KEYWORDS: {state['seo_keywords']}
    
    PRIME DIRECTIVES:
    1. No corporate fluff. No "In the ever-evolving landscape."
    2. Use precise technical vocabulary mixed with high-level philosophy.
    3. Include one bulleted list of "Hard Truths."
    4. Max 180 words.
    """
    draft = llm_alchemist.invoke(prompt).content.strip()
    
    # Basic cleanup
    draft = re.sub(r'^(Here is|Sure|The section).*?\n', '', draft, flags=re.I)
    
    return {"section_content": draft, "current_section": current_section}

# --- NODE 5: PROOFREADER GENIUS (The Syntax Sentinel) ---
def syntax_sentinel(state: AgentState):
    print(f"🧐 Syntax Sentinel: Auditing section {state['iteration'] + 1}...")
    
    content = state['section_content']
    
    # Forensic Audit
    prompt = f"""
    Audit this text for "Hallucinations" and "Fluff".
    Text: {content}
    
    If it contains clichés like "game-changer" or "revolutionize", reject it.
    If it is intellectually lazy, reject it.
    
    Return ONLY 'APPROVED' or 'REJECTED'.
    """
    verdict = llm_sovereign.invoke(prompt).content.strip().upper()
    
    if "REJECTED" in verdict and state.get('quality_checks', 0) < 2:
        print("   ❌ REJECTED. Demanding rewrite...")
        return {"quality_checks": state.get('quality_checks', 0) + 1}
    
    print("   ✅ APPROVED. Integrating to Matrix.")
    
    # Format the section
    formatted_section = f"\n\n## {state['current_section']}\n\n{content}\n"
    
    return {
        "full_draft": state['full_draft'] + formatted_section, 
        "iteration": state['iteration'] + 1,
        "quality_checks": 0 # Reset for next section
    }

# --- NODE 6: PUBLISHING KING (The Distribution Overlord) ---
def publishing_king(state: AgentState):
    print("👑 Publishing King: Finalizing Omni-Channel Distribution...")
    
    # 1. Generate The Visual Asset (via Pollinations.ai)
    timestamp = int(time.time())
    img_filename = f"apex-{timestamp}.png"
    img_dir = "assets/img"
    os.makedirs(img_dir, exist_ok=True)
    img_path = os.path.join(img_dir, img_filename)
    
    # Create the "Visual Concept"
    vis_prompt = f"Describe an abstract, avant-garde, hyper-realistic image for '{state['topic']}'. Dark, neon, cinematic. 15 words max."
    visual_concept = llm_alchemist.invoke(vis_prompt).content.strip().replace('"', '')
    
    encoded_prompt = urllib.parse.quote(f"{visual_concept}, 8k resolution, unreal engine 5 render, wide angle")
    gen_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&seed={timestamp}&nologo=true"
    
    try:
        requests.get(gen_url, timeout=30)
        # Pollinations saves directly via URL, but we download for persistence
        with open(img_path, "wb") as f:
            f.write(requests.get(gen_url).content)
        final_img_url = img_filename
        print(f"   🖼️ Asset Secured: {img_filename}")
    except:
        final_img_url = ""
        print("   ⚠️ Visual Asset Failed. Using fallback.")

    # 2. Final Polish (Key Takeaways)
    summary_prompt = f"Summarize this entire article into 3 'Prophetic Insights'. Article: {state['full_draft']}"
    takeaways = llm_sovereign.invoke(summary_prompt).content.strip()
    takeaways = re.sub(r'^(Here are|Sure).*?\n', '', takeaways, flags=re.I)
    
    final_content = f"> ### Prophetic Insights\n>\n{takeaways}\n\n&nbsp;\n\n{state['full_draft']}"
    
    return {"content": final_content, "image_url": final_img_url}

# --- CONDITIONAL ROUTING ---
def quality_router(state: AgentState) -> Literal["rewrite", "next", "finalize"]:
    # If rejected and under retry limit -> Rewrite
    if state.get('quality_checks', 0) > 0:
        return "rewrite"
    # If more sections needed -> Next
    if state['iteration'] < len(state['outline']):
        return "next"
    # Done -> Finalize
    return "finalize"

# --- GRAPH CONSTRUCTION ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("researcher", deep_data_diviner)
workflow.add_node("seo_strategist", seo_apex_strategist)
workflow.add_node("editor", master_editor)
workflow.add_node("writer", prompt_commander)
workflow.add_node("auditor", syntax_sentinel)
workflow.add_node("publisher", publishing_king)

# Set Edges
workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "seo_strategist")
workflow.add_edge("seo_strategist", "editor")
workflow.add_edge("editor", "writer")
workflow.add_edge("writer", "auditor")

# Conditional Logic from Auditor
workflow.add_conditional_edges(
    "auditor",
    quality_router,
    {
        "rewrite": "writer",
        "next": "writer",
        "finalize": "publisher"
    }
)

workflow.add_edge("publisher", END)
app = workflow.compile()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("🔮 THE SOVEREIGN INTELLIGENCE SYNDICATE IS ONLINE.")
    
    final_state = app.invoke({
        "field": "Artificial Intelligence & Robotics",
        "topic": "",
        "research_data": "",
        "seo_keywords": "",
        "outline": [],
        "current_section": "",
        "section_content": "",
        "full_draft": "",
        "iteration": 0,
        "quality_checks": 0,
        "image_url": ""
    })
    
    # Save File Logic
    now = datetime.datetime.now()
    today_date = now.strftime("%Y-%m-%d")
    clean_topic = final_state['topic'].replace('"', '').strip()
    slug = re.sub(r'[^a-z0-9]', '-', clean_topic.lower()).strip("-")[:50]
    
    # Final Formatting Scrub (The DevOps Safety Net)
    content = final_state['content']
    content = re.sub(r'##\s*(H2|Header|Section):?\s*', '## ', content, flags=re.I)
    
    img_path_str = f"/assets/img/{final_state['image_url']}" if final_state['image_url'] else "https://images.unsplash.com/photo-1518770660439-4636190af475?w=1200"
    
    post_md = f"""---
layout: post
title: "{clean_topic}"
date: {today_date} 09:00:00 +0200
categories: [Apex, Technology]
image:
  path: {img_path_str}
  alt: "{clean_topic}"
---

{content}"""

    os.makedirs("_posts", exist_ok=True)
    with open(f"_posts/{today_date}-{slug}.md", "w", encoding="utf-8") as f:
        f.write(post_md)
        
    print(f"🚀 PUBLISHED: {today_date}-{slug}.md")
