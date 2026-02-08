import os
import requests
import re
import time
import datetime
import urllib.parse
from google import genai
from google.genai import types
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS  # Changed: Direct import for stability

# --- 1. CONFIGURATION & STATE ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

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


# --- 2. NODES ---

def deep_data_diviner(state: AgentState):
    current_year = datetime.date.today().year
    print(f"🕵️ Researcher: Scouting {state['field']} breakthroughs...")
    query = (
        f"latest {state['field']} breakthroughs {current_year} "
        "research releases funding open-source models"
    )

    try:
        # Changed: Using DDGS context manager directly to avoid LangChain import errors
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=5)]
            raw_data = "\n".join(results)
    except Exception as e:
        print(f"⚠️ Search failed, using fallback. Error: {e}")
        raw_data = (
            f"Recent developments in {state['field']} spanning model releases, "
            "agentic workflows, and compute-efficient architectures from the last 90 days."
        )

    prompt = (
        f"Data: {raw_data}\n"
        "Identify the single most up-to-date technical niche topic focused on current AI developments. "
        "Return ONLY the title."
    )
    topic = llm_sovereign.invoke(prompt).content.strip().replace('"', "")
    return {"topic": topic, "research_data": raw_data}


def seo_apex_strategist(state: AgentState):
    prompt = f"Provide 5 SEO keywords for '{state['topic']}'. Comma-separated ONLY."
    keywords = llm_alchemist.invoke(prompt).content.strip()
    return {"seo_keywords": keywords}


def master_editor(state: AgentState):
    print(f"🏛️ Master Editor: Structuring narrative for {state['topic']}...")
    prompt = (
        f"Create a 4-section technical outline for: {state['topic']}. "
        "Return ONLY section titles, 1 per line."
    )
    raw_out = llm_sovereign.invoke(prompt).content
    sections = [line.strip() for line in raw_out.split("\n") if len(line.strip()) > 5]
    return {"outline": sections[:4], "iteration": 0, "full_draft": ""}


def prompt_commander(state: AgentState):
    current_section = state["outline"][state["iteration"]]
    iteration = state["iteration"]

    focus = "overview and current news" if iteration == 0 else "technical deep-dive and specific implementation details"

    print(f"👻 Writer: Drafting section {iteration + 1}: {current_section}...")
    prompt = f"""
    SECTION: {current_section}
    FOCUS: {focus}
    CONTEXT: {state['research_data']}
    CONSTRAINTS:
    - No headers, titles, or intros.
    - DO NOT repeat facts already mentioned in previous sections.
    - Start immediately with the content.
    - Use technical, professional language.
    - Emphasize recent AI developments from the last 12 months.
    """
    draft = llm_alchemist.invoke(prompt).content.strip()
    return {"section_content": draft, "current_section": current_section}


def syntax_sentinel(state: AgentState):
    """PERMANENT SOLUTION: Removes H2 tags and Duplicate Titles."""
    content = state["section_content"]
    section_title = state["current_section"].strip()

    # Remove any markdown headers (#) or literal 'h2:' tags produced by LLM
    clean = re.sub(r"(?i)^#+.*$", "", content, flags=re.MULTILINE)
    clean = re.sub(r"(?i)^h2:.*$", "", clean, flags=re.MULTILINE)

    lines = clean.split("\n")
    # Filter out lines that repeat the title and empty lines
    filtered_lines = [l for l in lines if l.strip().lower() != section_title.lower() and l.strip()]

    final_body = "\n\n".join(filtered_lines)
    # Wrap in clean H2 tags for Jekyll
    formatted_section = f"\n\n## {section_title}\n\n{final_body}\n"

    return {"full_draft": state["full_draft"] + formatted_section, "iteration": state["iteration"] + 1}


def _detect_image_extension(content_type: str) -> str:
    content_type = content_type.lower()
    if "image/webp" in content_type:
        return "webp"
    if "image/jpeg" in content_type or "image/jpg" in content_type:
        return "jpg"
    if "image/png" in content_type:
        return "png"
    return "png"


def _write_image(img_path: str, image_bytes: bytes) -> None:
    with open(img_path, "wb") as f:
        f.write(image_bytes)


def _try_gemini_image(prompt: str, model: str, img_path: str) -> bool:
    if not client:
        print("⚠️ GEMINI_API_KEY not set. Skipping Gemini image generation.")
        return False
    try:
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )
        images = getattr(response, "generated_images", None)
        if images:
            _write_image(img_path, images[0].image_bytes)
            return True
        print(f"⚠️ Gemini model {model} returned no images.")
    except Exception as e:
        print(f"⚠️ Gemini {model} image generation failed: {e}")
    return False


def _try_pollinations(prompt: str, img_basename: str, img_dir: str, seed: int) -> Optional[str]:
    encoded = urllib.parse.quote(prompt[:80])
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width=1280&height=720&nologo=true&seed={seed}"
    )
    try:
        res = requests.get(
            url,
            timeout=20,
            headers={"User-Agent": "ai-news-agent/1.0"},
        )
        if res.status_code != 200:
            print(f"⚠️ Pollinations returned status {res.status_code}.")
            return None
        content_type = res.headers.get("Content-Type", "").lower()
        if "image/" not in content_type:
            preview = res.text[:200].replace("\n", " ")
            print(f"⚠️ Pollinations returned non-image content: {content_type}. Preview: {preview}")
            return None
        img_ext = _detect_image_extension(content_type)
        img_filename = f"{img_basename}.{img_ext}"
        img_path = os.path.join(img_dir, img_filename)
        _write_image(img_path, res.content)
        return img_filename
    except Exception as e:
        print(f"⚠️ Pollinations image generation failed: {e}")
        return None


def _try_unsplash(keyword_source: str, img_basename: str, img_dir: str) -> Optional[str]:
    try:
        encoded = urllib.parse.quote(keyword_source[:60])
        res = requests.get(
            f"https://source.unsplash.com/1600x900/?{encoded}",
            timeout=20,
            headers={"User-Agent": "ai-news-agent/1.0"},
            allow_redirects=True,
        )
        if res.status_code != 200:
            print(f"⚠️ Unsplash returned status {res.status_code}.")
            return None
        content_type = res.headers.get("Content-Type", "").lower()
        if "image/" not in content_type:
            preview = res.text[:200].replace("\n", " ")
            print(f"⚠️ Unsplash returned non-image content: {content_type}. Preview: {preview}")
            return None
        img_ext = _detect_image_extension(content_type)
        img_filename = f"{img_basename}-unsplash.{img_ext}"
        img_path = os.path.join(img_dir, img_filename)
        _write_image(img_path, res.content)
        return img_filename
    except Exception as e:
        print(f"⚠️ Unsplash fallback failed: {e}")
        return None


def publishing_king(state: AgentState):
    print("👑 Publisher: Finalizing and Saving...")

    timestamp = int(time.time())
    img_basename = f"apex-{timestamp}"
    img_dir = "assets/img"
    os.makedirs(img_dir, exist_ok=True)

    prompt = (
        f"Futuristic professional tech visual for {state['topic']}, "
        "digital art style, high resolution"
    )

    img_filename: Optional[str] = None
    img_path = os.path.join(img_dir, f"{img_basename}.png")

    success = _try_gemini_image(prompt, "gemini-2.0-flash-exp-image-generation", img_path)
    if not success:
        success = _try_gemini_image(prompt, "imagen-3.0-generate-001", img_path)

    if success:
        img_filename = os.path.basename(img_path)
    else:
        img_filename = _try_pollinations(state["topic"], img_basename, img_dir, timestamp)

    if not img_filename:
        print("⚠️ Image generation failed. Falling back to Unsplash keyword image.")
        keyword_source = state.get("seo_keywords") or state["topic"]
        img_filename = _try_unsplash(keyword_source, img_basename, img_dir)

    if not img_filename:
        print("⚠️ Unsplash fallback failed. Continuing without a header image.")

    date_str = datetime.date.today().strftime("%Y-%m-%d")
    date_full = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S +0000")
    safe_slug = re.sub(r"[^a-z0-9-]", "", state["topic"].lower().replace(" ", "-"))[:50]

    post_path = os.path.join("_posts", f"{date_str}-{safe_slug}.md")
    os.makedirs("_posts", exist_ok=True)

    image_block = ""
    if img_filename:
        image_block = f"""image:\n  path: /assets/img/{img_filename}\n"""

    header = f"""---\ntitle: \"{state['topic']}\"\ndate: {date_full}\ncategories: [{state['field']}]\ntags: [{state['seo_keywords']}]\n{image_block}---\n\n"""
    final_markdown = header + state["full_draft"]

    with open(post_path, "w", encoding="utf-8") as f:
        f.write(final_markdown)

    print(f"✅ ARTICLE SAVED: {post_path}")
    return {"content": final_markdown, "image_url": img_filename or ""}


# --- 3. GRAPH & RUNTIME ---

def router(state: AgentState):
    return "finalize" if state["iteration"] >= len(state["outline"]) else "next"


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
    app.invoke({"field": "AI developments", "full_draft": "", "iteration": 0})
