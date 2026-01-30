import os
import requests
import random
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun

# 1. Define the Shared State (The Agent's "Brain")
class AgentState(TypedDict):
    field: str        # Broad area (e.g., "AI in Medicine")
    topic: str        # Specific selected headline
    research: str     # Raw search data
    outline: List[str]# Section titles
    content: str      # The growing article
    iteration: int    # Current section counter
    image_url: str    # URL of generated header

# 2. Setup Tools & Models
# Using Llama 3.3 70B for its superior logic and formatting adherence
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
search = DuckDuckGoSearchRun()

# --- THE TASK FORCE NODES ---

def trend_scout_node(state: AgentState):
    """Scans the web for the most 'viral' news in the given field."""
    print(f"📡 Trend Scout: Scanning {state['field']} for 2026 breakthroughs...")
    raw_news = search.run(f"latest trending news and breakthroughs in {state['field']} 2026")
    
    prompt = f"""Based on this news: {raw_news}
    Pick the SINGLE most impactful and SEO-rich story for a 2,000-word deep dive.
    Return ONLY the title of the article."""
    
    selected_topic = llm.invoke(prompt).content.strip().replace('"', '')
    print(f"🎯 Topic Selected: {selected_topic}")
    return {"topic": selected_topic}

def researcher_node(state: AgentState):
    """Performs deep-dive research on the chosen topic."""
    print(f"🕵️ Researcher: Deep-diving into '{state['topic']}'...")
    queries = [
        f"{state['topic']} technical details and facts 2026",
        f"{state['topic']} expert opinions and industry impact",
        f"{state['topic']} statistics and future predictions"
    ]
    results = [search.run(q) for q in queries]
    return {"research": "\n\n".join(results)}

def architect_node(state: AgentState):
    """Creates a 5-6 section SEO-optimized blueprint."""
    print("📐 Architect: Building 2026 SEO/AIO outline...")
    prompt = f"""Using this research: {state['research']}
    Create a 6-section outline for an article titled '{state['topic']}'.
    Each section should be a question-based H2 (e.g., 'How does X work?').
    Return ONLY the section titles, one per line."""
    
    response = llm.invoke(prompt).content
    sections = [s.strip() for s in response.split('\n') if len(s.strip()) > 5]
    return {"outline": sections}

def writer_node(state: AgentState):
    """Writes one section at a time to achieve massive length and detail."""
    current_section = state['outline'][state['iteration']]
    print(f"✍️ Writer: Drafting Section {state['iteration'] + 1} of {len(state['outline'])}...")
    
    prompt = f"""Write a comprehensive section for the heading: '## {current_section}'.
    STRICT 2026 AIO RULES:
    1. Answer-First: Start with a bolded, 1-sentence direct answer.
    2. Skimmable: Paragraphs must be max 3 sentences.
    3. Formatting: Use bullet points or numbered lists where possible.
    4. Depth: Use research context: {state['research']}.
    Length target: 400 words."""
    
    response = llm.invoke(prompt).content
    return {
        "content": state['content'] + "\n\n" + response, 
        "iteration": state['iteration'] + 1
    }

def aio_editor_node(state: AgentState):
    """Adds the 'Key Takeaways' box required for AI search snippets."""
    print("📋 AIO Editor: Creating the 'Key Takeaways' snippet...")
    prompt = f"""Based on this article: {state['content'][:2000]}
    Create a 'Key Takeaways' box.
    Format: Use a Markdown blockquote (>) with 3 bullet points.
    Goal: Summarize the article for an AI Search Engine overview."""
    
    box = llm.invoke(prompt).content
    full_article = f"# {state['topic']}\n\n{box}\n\n{state['content']}"
    return {"content": full_article}

def designer_node(state: AgentState):
    """Generates a high-quality visual for the article header."""
    print("🎨 Designer: Designing the visual identity...")
    prompt_gen = f"Describe a cinematic, futuristic 8k digital art piece for: {state['topic']}. Max 10 words."
    img_prompt = llm.invoke(prompt_gen).content.strip().replace(" ", "-")
    
    # Using Pollinations.ai for automated image generation
    image_url = f"https://image.pollinations.ai/prompt/{img_prompt}?width=1280&height=720&nologo=true&seed={random.randint(1,1000)}"
    
    header_img = f"![Featured Image]({image_url})\n\n"
    return {"content": header_img + state['content'], "image_url": image_url}

# --- THE GRAPH ARCHITECTURE ---

workflow = StateGraph(AgentState)

# Register nodes
workflow.add_node("trend_scout", trend_scout_node)
workflow.add_node("researcher", researcher_node)
workflow.add_node("architect", architect_node)
workflow.add_node("writer", writer_node)
workflow.add_node("aio_editor", aio_editor_node)
workflow.add_node("designer", designer_node)

# Connect nodes
workflow.set_entry_point("trend_scout")
workflow.add_edge("trend_scout", "researcher")
workflow.add_edge("researcher", "architect")
workflow.add_edge("architect", "writer")

# Logic: Loop back to 'writer' until the outline is finished
def should_continue(state: AgentState):
    if state['iteration'] < len(state['outline']):
        return "writer"
    return "aio_editor"

workflow.add_conditional_edges("writer", should_continue)
workflow.add_edge("aio_editor", "designer")
workflow.add_edge("designer", END)

app = workflow.compile()

# --- EXECUTION ---

if __name__ == "__main__":
    # Change this field to whatever you want the agent to 'scout' for news
    MY_FIELD = "Artificial Intelligence and Robotics"
    
    print(f"🚀 Starting AI Newsroom for: {MY_FIELD}")
    
    final_state = app.invoke({
        "field": MY_FIELD,
        "topic": "",
        "content": "",
        "iteration": 0
    })
    
    # Save the mega-article
    os.makedirs("content", exist_ok=True)
    filename = "content/latest_news.md"
    with open(filename, "w") as f:
        f.write(final_state['content'])
    
    print(f"✅ Success! 2,000+ word article published with custom imagery to {filename}")
