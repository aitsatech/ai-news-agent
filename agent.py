import os
import requests
import re
import time
import urllib.parse
from google import genai
from google.genai import types 
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# --- CONFIGURATION ---
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
    print(f"🕵️ Researcher: Scouting technical niche...")
    query = f"latest {state['field']} breakthroughs 2026"
    raw_data = DuckDuckGoSearchRun().run(query)
    prompt = f"Data: {raw_data}\nIdentify the single most technical niche topic. Return ONLY the title."
    topic = llm_sovereign.invoke(prompt).content.strip().replace('"', '')
    return {"topic": topic, "research_data": raw_data}

def seo_apex_strategist(state: AgentState):
    prompt = f"Provide 5 SEO keywords for '{state['topic']}'. Comma-separated ONLY."
    keywords = llm_alchemist.invoke(prompt).content.strip()
    return {"seo_keywords": keywords}

def master_editor(state: AgentState):
    print("🏛️ Master Editor: Structuring narrative...")
    prompt = f"Create a 4-section technical outline for: {state['topic']}. Return ONLY section titles, 1 per line."
    raw_out = llm_sovereign.invoke(prompt).content
    sections = [line.strip() for line in raw_out.split('\n') if len(line.strip()) > 5]
    return {"outline": sections[:4], "iteration": 0, "full_draft": ""}

def prompt_commander(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"👻 Writer: Drafting {current_section}...")
    prompt = f"""
    SECTION: {current_section}
    CONTEXT: {state['research_data']}
    CONSTRAINTS: 
    - No headers or titles. 
    - Start immediately with the content.
    - No introductory phrases.
    """
    draft = llm_alchemist.invoke(prompt).content.strip()
    return {"section_content": draft, "current_section": current_section}

def syntax_sentinel(state: AgentState):
    """
    PERMANENT SOLUTION: Hard-strips H2 tags and Duplicate Lines.
    """
    content = state['section_content']
    section_title = state['current_section'].strip()
    
    # 1. Regex Strip: Removes lines starting with # or "H2:"
    clean = re.sub(r'(?i)^#+.*$', '', content, flags=re.MULTILINE)
    clean = re.sub(r'(?i)^h2:.*$', '', clean, flags=re.MULTILINE)
    
    # 2. Logic Strip: Removes lines that match the title exactly
    lines = clean.split('\n')
    filtered_lines = [l for l in lines if l.strip().lower() != section_title.lower() and l.strip()]
    
    final_body = "\n\n".join(filtered_lines)
    
    formatted_section = f"\n\n## {section_title}\n\n{final_body}\n"
    
    return {
        "full_draft": state['full_draft'] + formatted_section, 
        "iteration": state['iteration'] + 1
    }

def publishing_king(state: AgentState):
    print("👑 Publisher: Executing Image Failover...")
    
    timestamp = int(time.time())
    img_filename = f"apex-{timestamp}.png"
    img_path = os.path.join("assets/img", img_filename)
    os.makedirs("assets/img", exist_ok=True)
    
    # Generate a shorter prompt for fallbacks to avoid URL errors
    img_prompt = f"Cinematic tech visual for {state['topic']}, 8k, futuristic."
    short_prompt = f"Technology {state['topic']} futuristic"
    
    success = False
    
    # --- PROVIDER 1: NANO BANANA (GEMINI IMAGEN) ---
    print("   🎨 Attempting Nano Banana (Imagen 3)...")
    try:
        # We try the generic 'imagen-3.0-generate-001'
        response = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=img_prompt,
            config=types.GenerateImagesConfig(number_of_images=1)
        )
        if response.generated_images:
            with open(img_path, "wb") as f:
                f.write(response.generated_images[0].image_bytes)
            success = True
            print("   ✅ Image saved via Nano Banana")
    except Exception as e:
        # Clean error handling to avoid scary logs
        error_msg = str(e)
        if "404" in error_msg:
            print("   ⚠️ Nano Banana skipped: Model not enabled for this API Key.")
        else:
            print(f"   ⚠️ Nano Banana skipped: {error_msg[:100]}...")

    # --- PROVIDER 2: POLLINATIONS (AI FALLBACK) ---
    if not success:
        print("   🎨 Attempting Pollinations...")
        # Use simpler prompt to prevent URL length errors
        encoded = urllib.parse.quote(short_prompt) 
        poll_url = f"https://image.pollinations.ai/prompt/{encoded}?nologo=true&seed={timestamp}&width=1280&height=720"
        
        try:
            # Added User-Agent to look like a browser
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            res = requests.get(poll_url, headers=headers, timeout=25)
            
            if res.status_code == 200 and len(res.content) > 1000:
                with open(img_path, "wb") as f:
                    f.write(res.content)
                success = True
                print("   ✅ Image saved via Pollinations")
            else:
                print(f"   ⚠️ Pollinations failed (Status: {res.status_code})")
        except Exception as e:
            print(f"   ⚠️ Pollinations error: {e}")

    # --- PROVIDER 3: PICKSUM (STATIC FALLBACK) ---
    if not success:
        print("   🎨 Attempting Static Fallback...")
        try:
            res = requests.get(f"https://picsum.photos/seed/{timestamp}/1280/720", timeout=15)
            with open(img_path, "wb") as f:
                f.write(res.content)
            success = True
            print("   ✅ Image saved via Picsum")
        except:
            print("   ❌ CRITICAL: All image providers failed.")

    return {"content": state['full_draft'], "image_url": img_filename}

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
    app.invoke({"field": "AI & Robotics", "full_draft": "", "iteration": 0})
