import os
import requests
import random
import datetime
import re
from PIL import Image
from io import BytesIO
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from google import genai
from google.genai import types

# Initialize Gemini Client for Nano Banana
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

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
    prompt = (
        f"Create a 4-section outline for '{state['topic']}'. "
        "Return ONLY the titles, one per line. Do NOT include labels like 'H2', 'Section', or numbers."
    )
    raw_sections = llm_strategy.invoke(prompt).content.split('\n')
    sections = [re.sub(r'^(H2|Section|Step|Title)\s*:?\s*', '', s, flags=re.I).strip() for s in raw_sections if len(s.strip()) > 5]
    return {"outline": sections[:4]}

def writer_node(state: AgentState):
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer: Drafting {current_section}...")
    prompt = f"""Write a section for: {current_section}. 
    Research: {state['research']}. 
    RULES: 
    1. DO NOT repeat the title or include any headers in your response.
    2. Bold the first sentence. 
    3. Max 250 words. 
    4. Use 1 bulleted list."""
    
    res = llm_writer.invoke(prompt).content.strip()
    # Manually adding header ensures no H2 tags or duplication
    section_md = f"\n\n## {current_section}\n\n{res}"
    return {"content": state['content'] + section_md, "iteration": state['iteration'] + 1}

def aio_editor_node(state: AgentState):
    print("📋 AIO Editor: Drafting the 'Key Takeaways' box...")
    prompt = f"Summarize this in 3 bullet points: {state['content'][:1000]}."
    box = llm_strategy.invoke(prompt).content
    return {"content": "> ### Key Takeaways\n>\n" + box + "\n\n" + state['content']}

def designer_node(state: AgentState):
    print("🎨 Designer: Generating high-fidelity header with Nano Banana...")
    img_prompt = (
        f"A cinematic, high-quality professional header image for a news article about {state['topic']}. "
        "Style: Futuristic digital art, clean, 8k resolution, vibrant lighting. No messy text."
    )
    
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[img_prompt],
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(aspect_ratio="16:9")
        )
    )

    os.makedirs("assets/img", exist_ok=True)
    img_filename = f"header-{datetime.datetime.now().strftime('%Y%m%d%H%M')}.png"
    img_path = f"assets/img/{img_filename}"
    
    for part in response.parts:
        if part.inline_data:
            image = Image.open(BytesIO(part.inline_data.data))
            image.save(img_path)
    
    image_md = f"![Header Image](/{img_path})\n\n"
    return {"content": image_md + state['content'], "image_url": img_path}

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
    
    # Strict regex to remove any "H2" text or duplicate headers generated by LLM
    clean_content = final_state['content'].replace("## H2", "##").replace("H2 ", "")
    clean_content = re.sub(r'\n{3,}', '\n\n', clean_content) 
    
    # Simplified front_matter to prevent triple-quote errors
    fm_lines = [
        "---",
        "layout: post",
        f'title: "{final_state["topic"]}"',
        f"date: {today} 12:00:00 +0200",
        "categories: [AI, News]",
        "tags: [ai, robotics, future]",
        "---",
        "",
        ""
    ]
    front_matter = "\n".join(fm_lines)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(front_matter + clean_content)
    
    print(f"✅ Success: Published {filename}")
