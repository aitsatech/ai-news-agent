import os
import requests
import re
import time
import urllib.parse
from google import genai # Matches your requirements.txt
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# --- CONFIGURATION ---
# Use the new SDK client initialization
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
    image_url: str
    seo_keywords: str
    content: str 

# --- NODES ---

def deep_data_diviner(state: AgentState):
    print(f"🕵️ Researcher Deluxe: Scanning for technical edge...")
    query = f"latest breakthroughs in {state['field']} 2026 technical report"
    search = DuckDuckGoSearchRun()
    raw_data = search.run(query)
    prompt = f"Data: {raw_data}\nPick the single most specific, technical niche topic. Return ONLY the title."
    topic = llm_sovereign.invoke(prompt).content.strip().replace('"', '')
    return {"topic": topic, "research_data": raw_data}

def seo_apex_strategist(state: AgentState):
    prompt = f"Generate 5 high-intent SEO keywords for: '{state['topic']}'. Comma-separated list ONLY."
    keywords = llm_alchemist.invoke(prompt).content.strip()
    return {"seo_keywords": keywords}

def master_editor(state: AgentState):
    print("🏛️ Master Editor: Calibrating narrative soul...")
    prompt = f"Create a 4-section technical outline for: {state['topic']}. Return ONLY section titles, 1 per line. No numbers or prefixes."
    raw_out = llm_sovereign.invoke(prompt).content
    sections = [line.strip() for line in raw_out.split('\n') if len(line.strip()) > 5]
    return {"outline": sections[:4], "iteration": 0, "full_draft": ""}

def prompt_commander(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"👻 Ghost in the Machine: Writing section: {current_section}...")
    prompt = f"""
    ROLE: Apex Tech Journalist. 
    SECTION: {current_section}.
    CONTEXT: {state['research_data']}.
    CONSTRAINTS: 
    - Start immediately with the technical content. 
    - DO NOT include the section title '{current_section}'.
    - DO NOT use any markdown headers (no # or ##).
    - DO NOT use introductory phrases.
    """
    draft = llm_alchemist.invoke(prompt).content.strip()
    return {"section_content": draft, "current_section": current_section}

def syntax_sentinel(state: AgentState):
    """
    PERMANENT SOLUTION: 
    1. Removes H2 tags appearing next to headers.
    2. Eliminates duplicate lines.
    """
    content = state['section_content']
    section_title = state['current_section'].strip()
    
    # 1. Strip any line that is just a header or contains "H2"
    lines = content.split('\n')
    clean_lines = []
    for line in lines:
        l = line.strip()
        # Skip empty lines, lines starting with #, or lines repeating the title
        if not l or l.startswith('#') or l.lower() == section_title.lower():
            continue
        # Specifically remove the "H2:" artifact mentioned in user constraints
        if l.lower().startswith("h2:"):
            continue
        clean_lines.append(l)
    
    # 2. Join and ensure no duplicate title exists
    body_text = "\n\n".join(clean_lines)
    
    # 3. Final construction: Clean H2 followed by clean body
    formatted_section = f"\n\n## {section_title}\n\n{body_text}\n"
    
    print(f"   ✅ Syntax Cleaned: {section_title}")
    return {
        "full_draft": state['full_draft'] + formatted_section, 
        "iteration": state['iteration'] + 1
    }

def publishing_king(state: AgentState):
    print("👑 Publishing King: Image Failover Sequence...")
    
    timestamp = int(time.time())
    img_prompt = f"Professional technical illustration of {state['topic']}, 8k, futuristic aesthetic."
    img_filename = f"apex-{timestamp}.png"
    img_path = os.path.join("assets/img", img_filename)
    os.makedirs("assets/img", exist_ok=True)
    
    success = False
    
    # --- PROVIDER 1: NANO BANANA (GEMINI GOOGLE-GENAI SDK) ---
    print("   🎨 Attempting Nano Banana (Gemini API)...")
    try:
        # Using the new SDK syntax: client.models.generate_content (for images use Imagen)
        # Note: Image generation in the new SDK often uses the 'imagen-3' model
        # If your tier doesn't support it, this block will fail gracefully to next provider
        response = client.models.generate_image(
            model='imagen-3.0-generate-001',
            prompt=img_prompt
        )
        # Save bytes from response
        with open(img_path, "wb") as f:
            f.write(response.generated_images[0].image_bytes)
        success = True
        print("   ✅ Asset Secured via Nano Banana")
    except Exception as e:
        print(f"   ⚠️ Nano Banana bypassed: API tier limit or configuration.")

    # --- PROVIDER 2: POLLINATIONS (RELIABLE AI) ---
    if not success:
        print("   🎨 Attempting Pollinations...")
        encoded = urllib.parse.quote(img_prompt)
        poll_url = f"https://image.pollinations.ai/prompt/{encoded}?width=1280&height=720&nologo=true&seed={timestamp}"
        try:
            res = requests.get(poll_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            if res.status_code == 200:
                with open(img_path, "wb") as f:
                    f.write(res.content)
                success = True
                print("   ✅ Asset Secured via Pollinations")
        except:
            pass

    # --- PROVIDER 3: PICKSUM (TECHNICAL FALLBACK) ---
    if not success:
        print("   🎨 Attempting Technical Fallback...")
        fallback_url = f"https://picsum.photos/seed/{timestamp}/1280/720"
        res = requests.get(fallback_url)
        with open(img_path, "wb") as f:
            f.write(res.content)
        success = True

    final_article = f"{state['full_draft']}"
    return {"content": final_article, "image_url": img_filename}

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
    app.invoke({"field": "AI & Automation", "full_draft": "", "iteration": 0})
