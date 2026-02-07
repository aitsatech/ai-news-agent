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

# Image Provider Configurations
IMAGE_PROVIDERS = [
    {"name": "Nano Banana", "url": "https://api.nano-banana.com/v1/generate?prompt={prompt}&width=1280&height=720"},
    {"name": "Pollinations", "url": "https://image.pollinations.ai/prompt/{prompt}?width=1280&height=720&nologo=true&nofeed=true"},
    {"name": "Herc AI", "url": "https://herc.ai/api/image?prompt={prompt}"} # Third Fallback Slot
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
    quality_checks: int
    image_url: str
    seo_keywords: str
    content: str 

# --- NODES ---

def deep_data_diviner(state: AgentState):
    print(f"🕵️ Researcher Deluxe: Initiating scan for '{state['field']}'...")
    query = f"cutting edge breakthroughs {state['field']} technical 2026"
    raw_data = DuckDuckGoSearchRun().run(query)
    prompt = f"From this data: {raw_data}\nPick the most complex, niche topic. Return ONLY the title."
    topic = llm_sovereign.invoke(prompt).content.strip().replace('"', '')
    print(f"   ↳ Topic Identified: {topic}")
    return {"topic": topic, "research_data": raw_data}

def seo_apex_strategist(state: AgentState):
    print("🌐 SEO Sovereign: Mapping neural entities...")
    prompt = f"Generate 5 high-intent SEO keywords for: '{state['topic']}'. Comma-separated list ONLY."
    keywords = llm_alchemist.invoke(prompt).content.strip()
    return {"seo_keywords": keywords}

def master_editor(state: AgentState):
    print("🏛️ Master Editor: Calibrating narrative soul...")
    prompt = f"Create a 4-section intellectual outline for: {state['topic']}. Return ONLY section titles, 1 per line."
    raw_out = llm_sovereign.invoke(prompt).content
    sections = [line.strip() for line in raw_out.split('\n') if len(line.strip()) > 5]
    return {"outline": sections[:4], "iteration": 0, "full_draft": ""}

def prompt_commander(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"👻 Ghost in the Machine: Writing section: {current_section}...")
    prompt = f"""
    ROLE: Apex Tech Journalist. SECTION: {current_section}.
    CONTEXT: {state['research_data']}. KEYWORDS: {state['seo_keywords']}.
    PRIME DIRECTIVES: No titles, no H2 tags, no fluff. Just ruthless analysis. Max 180 words.
    """
    draft = llm_alchemist.invoke(prompt).content.strip()
    return {"section_content": draft, "current_section": current_section}

def syntax_sentinel(state: AgentState):
    content = state['section_content']
    # CLEANUP: Remove any H2 markers or duplicate section names the LLM might have hallucinated
    clean_content = re.sub(r'^(#+.*|' + re.escape(state['current_section']) + r'[:\-]*)\n*', '', content, flags=re.I).strip()
    
    formatted_section = f"\n\n## {state['current_section']}\n\n{clean_content}\n"
    print(f"   ✅ Approved & Cleaned: Section {state['iteration'] + 1}")
    return {
        "full_draft": state['full_draft'] + formatted_section, 
        "iteration": state['iteration'] + 1
    }

def publishing_king(state: AgentState):
    print("👑 Publishing King: Executing Multi-Provider Failover Loop...")
    
    timestamp = int(time.time())
    vis_prompt = f"Futuristic technology, {state['topic']}, cinematic lighting, 8k, hyper-detailed."
    encoded_prompt = urllib.parse.quote(vis_prompt)
    
    img_filename = f"apex-{timestamp}.png"
    img_path = os.path.join("assets/img", img_filename)
    os.makedirs("assets/img", exist_ok=True)
    
    final_img_url = ""
    success = False
    
    while not success:
        for provider in IMAGE_PROVIDERS:
            try:
                print(f"   🎨 Attempting {provider['name']}...")
                target_url = provider['url'].format(prompt=encoded_prompt)
                response = requests.get(target_url, timeout=45)
                
                if response.status_code == 200:
                    with open(img_path, "wb") as f:
                        f.write(response.content)
                    if os.path.getsize(img_path) > 5000:
                        final_img_url = img_filename
                        print(f"   ✅ Asset Secured via {provider['name']}")
                        success = True
                        break
                else:
                    print(f"   ⚠️ {provider['name']} busy (Status {response.status_code}). Trying next...")
            except Exception as e:
                print(f"   ❌ Error with {provider['name']}: {e}")
        
        if not success:
            print("   ⏳ All providers failed. Cooling down for 10s before restart...")
            time.sleep(10)

    # Prophetic Insights Cleanup
    summary_prompt = f"Synthesize 3 Prophetic Insights for 2026 based on: {state['full_draft']}"
    insights = llm_sovereign.invoke(summary_prompt).content.strip()
    # REGEX: Strip "Based on...", "Here are...", "Synthesis:" etc.
    insights = re.sub(r'^.*?(synthesis|insights|here are|sure|based on).*?:\s*', '', insights, flags=re.I | re.S).strip()
    
    final_content = f"> ### Prophetic Insights\n>\n{insights}\n\n{state['full_draft']}"
    
    # FINAL H2 SENTINEL: Remove any lingering "H2:" or "Header:" labels
    final_content = re.sub(r'##\s*(H2|Header|Section|Title):?\s*', '## ', final_content, flags=re.I)
    
    return {"content": final_content, "image_url": final_img_url}

# --- GRAPH CONSTRUCTION ---
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
    res = app.invoke({"field": "Artificial Intelligence & Robotics", "full_draft": "", "iteration": 0})
    
    # Save File logic...
    filename = f"_posts/{datetime.date.today()}-article.md"
    # (Rest of your post-processing and file writing code)
    print(f"🚀 PUBLISHED: {filename}")
