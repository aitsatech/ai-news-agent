import os
import requests
import datetime
import re
import time
import urllib.parse
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# --- CONFIGURATION ---
llm_sovereign = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1)
llm_alchemist = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)

# Optimized Image Provider List
IMAGE_PROVIDERS = [
    {"name": "Nano Banana", "url": "https://api.nano-banana.com/v1/generate?prompt={prompt}&width=1280&height=720"}, 
    {"name": "Pollinations", "url": "https://image.pollinations.ai/prompt/{prompt}?width=1280&height=720&nologo=true&seed={seed}"},
    {"name": "Herc AI", "url": "https://herc.ai/api/image?prompt={prompt}"},
    {"name": "Stable Fallback", "url": "https://loremflickr.com/1280/720/{keyword}"} # Guaranteed to work
]

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
    print(f"🕵️ Researcher Deluxe: Initiating deep-scan...")
    query = f"latest breakthroughs in {state['field']} 2026 technical report"
    raw_data = DuckDuckGoSearchRun().run(query)
    prompt = f"Data: {raw_data}\nPick the single most specific, technical niche topic. Return ONLY the title."
    topic = llm_sovereign.invoke(prompt).content.strip().replace('"', '')
    print(f"   ↳ Topic Identified: {topic}")
    return {"topic": topic, "research_data": raw_data}

def seo_apex_strategist(state: AgentState):
    prompt = f"Generate 5 high-intent SEO keywords for: '{state['topic']}'. Comma-separated list ONLY."
    keywords = llm_alchemist.invoke(prompt).content.strip()
    return {"seo_keywords": keywords}

def master_editor(state: AgentState):
    print("🏛️ Master Editor: Calibrating narrative soul...")
    prompt = f"Create a 4-section technical outline for: {state['topic']}. Return ONLY section titles, 1 per line. No numbers."
    raw_out = llm_sovereign.invoke(prompt).content
    sections = [line.strip() for line in raw_out.split('\n') if len(line.strip()) > 5]
    return {"outline": sections[:4], "iteration": 0, "full_draft": ""}

def prompt_commander(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"Ghost in the Machine: Writing section: {current_section}...")
    prompt = f"""
    ROLE: Apex Tech Journalist. 
    SECTION: {current_section}.
    CONTEXT: {state['research_data']}.
    CONSTRAINTS: Start immediately with the content. NO TITLES. NO H2 TAGS. NO INTROS. 
    Max 150 words. Focus on raw technical insight.
    """
    draft = llm_alchemist.invoke(prompt).content.strip()
    return {"section_content": draft, "current_section": current_section}

def syntax_sentinel(state: AgentState):
    content = state['section_content']
    
    # PERMANENT FIX FOR H2 & DUPLICATES:
    # 1. Remove anything that looks like a Markdown header (#, ##, ###)
    # 2. Remove the section title if the AI repeated it
    clean_content = re.sub(r'^#+.*$', '', content, flags=re.MULTILINE)
    clean_content = re.sub(re.escape(state['current_section']), '', clean_content, flags=re.IGNORECASE).strip()
    
    # Construct the final block with exactly one H2
    formatted_section = f"\n\n## {state['current_section']}\n\n{clean_content}\n"
    
    print(f"   ✅ Approved: Section {state['iteration'] + 1}")
    return {
        "full_draft": state['full_draft'] + formatted_section, 
        "iteration": state['iteration'] + 1
    }

def publishing_king(state: AgentState):
    print("👑 Publishing King: Executing Circuit-Breaker Failover...")
    
    timestamp = int(time.time())
    encoded_prompt = urllib.parse.quote(f"Futuristic technology, {state['topic']}, cinematic, 8k")
    keyword_fallback = urllib.parse.quote(state['topic'].split()[0])
    
    img_filename = f"apex-{timestamp}.png"
    img_path = os.path.join("assets/img", img_filename)
    os.makedirs("assets/img", exist_ok=True)
    
    success = False
    max_cycles = 3
    
    for cycle in range(max_cycles):
        if success: break
        print(f"   🔄 Cycle {cycle + 1} of {max_cycles}...")
        
        for provider in IMAGE_PROVIDERS:
            try:
                # Format URL based on provider requirements
                if provider['name'] == "Stable Fallback":
                    target_url = provider['url'].format(keyword=keyword_fallback)
                elif provider['name'] == "Pollinations":
                    target_url = provider['url'].format(prompt=encoded_prompt, seed=timestamp)
                else:
                    target_url = provider['url'].format(prompt=encoded_prompt)

                print(f"   🎨 Trying {provider['name']}...")
                response = requests.get(target_url, timeout=15)
                
                if response.status_code == 200 and len(response.content) > 5000:
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                    print(f"   ✅ Asset Secured via {provider['name']}")
                    success = True
                    break
            except Exception:
                continue # Silent fail to next provider
        
        if not success:
            time.sleep(5) # Short breather between cycles

    # Final cleanup of "Prophetic Insights" (Removing conversational fluff)
    insight_prompt = f"Synthesize 3 Prophetic Insights for 2026 from: {state['full_draft']}"
    insights = llm_sovereign.invoke(insight_prompt).content.strip()
    insights = re.sub(r'^(Sure|Based on|Here are|Insights).*?:\s*', '', insights, flags=re.I | re.S).strip()
    
    final_content = f"> ### Prophetic Insights\n>\n{insights}\n\n{state['full_draft']}"
    
    # Final sweep for any lingering "H2:" text strings
    final_content = re.sub(r'##\s*(H2|Header|Title):?\s*', '## ', final_content, flags=re.I)
    
    return {"content": final_content, "image_url": img_filename if success else ""}

# --- GRAPH ---
def router(state: AgentState):
    return "finalize" if state['iteration'] >= len(state['outline']) else "next"

workflow = StateGraph(AgentState)
workflow.add_node("researcher", deep_data_diviner)
workflow.add_node("seo", seo_apex_strategist)
workflow.add_node("editor", master_editor)
workflow.add_node("writer", prompt_commander)
workflow.add_node("auditor", syntax_sentinel)
workflow.add_node("publisher", publishing_king)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "seo")
workflow.add_edge("seo", "editor")
workflow.add_edge("editor", "writer")
workflow.add_edge("writer", "auditor")
workflow.add_conditional_edges("auditor", router, {"next": "writer", "finalize": "publisher"})
workflow.add_edge("publisher", END)
app = workflow.compile()

if __name__ == "__main__":
    print("🔮 THE SOVEREIGN INTELLIGENCE SYNDICATE IS ONLINE.")
    app.invoke({"field": "Artificial Intelligence & Robotics", "full_draft": "", "iteration": 0})
