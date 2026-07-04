import os
import re
import glob
import time
import random
import difflib
import datetime
import urllib.parse

import requests
from google import genai
from google.genai import types
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from duckduckgo_search import DDGS  # Direct import for stability

# --- 1. CONFIGURATION & STATE ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Slightly higher temperature so topic selection isn't near-deterministic
llm_sovereign = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.4)
llm_alchemist = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)

# Rotating subfields so the search query (and therefore the topic) varies
# even on days when web search fails and we fall back to canned context.
SUBFIELDS = [
    "large language models",
    "AI agents and agentic workflows",
    "computer vision",
    "robotics and embodied AI",
    "reinforcement learning",
    "AI hardware and chips",
    "multimodal AI",
    "AI safety and alignment",
    "generative AI / diffusion models",
    "edge AI and on-device inference",
    "AI in healthcare",
    "open-source AI models",
]

# Phrases that indicate the LLM punted instead of returning a real topic.
BAD_TOPIC_MARKERS = [
    "none found",
    "no specific",
    "the text only mentions",
    "i could not find",
    "i cannot find",
    "no clear topic",
    "not enough information",
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


# --- 2. HELPERS: TOPIC MEMORY / DEDUPLICATION ---

def _load_past_titles(posts_dir: str = "_posts", limit: int = 60) -> List[str]:
    """Scan existing post front matter for titles so we never repeat a topic."""
    titles = []
    for path in sorted(glob.glob(os.path.join(posts_dir, "*.md")))[-limit:]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read(2000)  # front matter is always near the top
            match = re.search(r'^title:\s*"(.+?)"\s*$', text, flags=re.MULTILINE)
            if match:
                titles.append(match.group(1).strip())
        except Exception:
            continue
    return titles


def _is_duplicate_topic(topic: str, past_titles: List[str], threshold: float = 0.6) -> bool:
    topic_norm = topic.strip().lower()
    if not topic_norm:
        return True
    for past in past_titles:
        ratio = difflib.SequenceMatcher(None, topic_norm, past.strip().lower()).ratio()
        if ratio >= threshold:
            return True
    return False


def _is_bad_topic(topic: str) -> bool:
    if not topic or len(topic) < 8:
        return True
    lowered = topic.lower()
    return any(marker in lowered for marker in BAD_TOPIC_MARKERS)


# --- 3. NODES ---

def deep_data_diviner(state: AgentState):
    # Rotate the subfield deterministically by day so repeated failed
    # searches don't all collapse onto the same generic fallback text.
    day_index = datetime.date.today().toordinal()
    subfield = SUBFIELDS[day_index % len(SUBFIELDS)]

    current_year = datetime.date.today().year
    print(f"🕵️ Researcher: Scouting {subfield} breakthroughs...")
    query = f"latest {subfield} breakthroughs {current_year} research releases funding open-source models"

    try:
        with DDGS() as ddgs:
            results = [r["body"] for r in ddgs.text(query, max_results=5)]
            raw_data = "\n".join(results).strip()
            if not raw_data:
                raise ValueError("DDGS returned no results")
    except Exception as e:
        print(f"⚠️ Search failed, using fallback. Error: {e}")
        raw_data = (
            f"Recent developments in {subfield} spanning model releases, "
            f"agentic workflows, and compute-efficient architectures from the last 90 days. "
            f"(fallback context, day {day_index})"
        )

    past_titles = _load_past_titles()
    exclusion_text = ""
    if past_titles:
        exclusion_text = (
            "Do NOT choose any of these already-covered topics or close variants of them:\n- "
            + "\n- ".join(past_titles)
            + "\n"
        )

    topic = ""
    for attempt in range(4):
        prompt = (
            f"Focus area: {subfield}\n"
            f"Data: {raw_data}\n"
            f"{exclusion_text}"
            "Identify ONE specific, current, technical niche topic in this focus area. "
            "It must be a concrete article title, not a meta-comment about the data. "
            "Return ONLY the title, nothing else."
        )
        candidate = llm_sovereign.invoke(prompt).content.strip().strip('"')

        if _is_bad_topic(candidate):
            print(f"⚠️ Attempt {attempt + 1}: rejected bad/empty topic: {candidate!r}")
            continue
        if _is_duplicate_topic(candidate, past_titles):
            print(f"⚠️ Attempt {attempt + 1}: rejected duplicate-ish topic: {candidate!r}")
            continue

        topic = candidate
        break

    if not topic:
        # Guaranteed-unique last resort: timestamp the subfield itself.
        topic = f"{subfield.title()}: Weekly Developments Roundup ({datetime.date.today().isoformat()})"
        print(f"⚠️ Falling back to guaranteed-unique topic: {topic}")

    return {"topic": topic, "research_data": raw_data, "field": subfield}


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
    """Removes H2 tags and Duplicate Titles."""
    content = state["section_content"]
    section_title = state["current_section"].strip()

    clean = re.sub(r"(?i)^#+.*$", "", content, flags=re.MULTILINE)
    clean = re.sub(r"(?i)^h2:.*$", "", clean, flags=re.MULTILINE)

    lines = clean.split("\n")
    filtered_lines = [l for l in lines if l.strip().lower() != section_title.lower() and l.strip()]

    final_body = "\n\n".join(filtered_lines)
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


def _try_gemini_imagen(prompt: str, img_path: str) -> bool:
    """
    Only imagen-3.0-generate-001 actually supports client.models.generate_images().
    This requires a billing-enabled key; on a free-tier key it will fail, which
    is fine -- we just fall through to the next option.
    """
    if not client:
        print("⚠️ GEMINI_API_KEY not set. Skipping Gemini image generation.")
        return False
    try:
        response = client.models.generate_images(
            model="imagen-3.0-generate-001",
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),
        )
        images = getattr(response, "generated_images", None)
        if images:
            _write_image(img_path, images[0].image_bytes)
            return True
        print("⚠️ Gemini Imagen returned no images.")
    except Exception as e:
        print(f"⚠️ Gemini Imagen generation failed: {e}")
    return False


def _try_pollinations(prompt: str, img_basename: str, img_dir: str, seed: int, retries: int = 2) -> Optional[str]:
    encoded = urllib.parse.quote(prompt[:80])
    url = (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width=1280&height=720&nologo=true&seed={seed}"
    )
    for attempt in range(retries):
        try:
            res = requests.get(
                url,
                timeout=30,
                headers={"User-Agent": "ai-news-agent/1.0"},
            )
            if res.status_code != 200:
                print(f"⚠️ Pollinations returned status {res.status_code} (attempt {attempt + 1}).")
                time.sleep(2)
                continue
            content_type = res.headers.get("Content-Type", "").lower()
            if "image/" not in content_type:
                preview = res.text[:200].replace("\n", " ")
                print(f"⚠️ Pollinations returned non-image content: {content_type}. Preview: {preview}")
                time.sleep(2)
                continue
            img_ext = _detect_image_extension(content_type)
            img_filename = f"{img_basename}.{img_ext}"
            img_path = os.path.join(img_dir, img_filename)
            _write_image(img_path, res.content)
            return img_filename
        except Exception as e:
            print(f"⚠️ Pollinations image generation failed (attempt {attempt + 1}): {e}")
            time.sleep(2)
    return None


def _try_picsum(img_basename: str, img_dir: str, seed: int) -> Optional[str]:
    """
    Reliable, always-on placeholder fallback. source.unsplash.com was
    deprecated years ago and always fails -- picsum.photos is a live,
    keyless service that guarantees we never publish without an image.
    """
    try:
        res = requests.get(
            f"https://picsum.photos/seed/{seed}/1280/720",
            timeout=20,
            headers={"User-Agent": "ai-news-agent/1.0"},
            allow_redirects=True,
        )
        if res.status_code != 200:
            print(f"⚠️ Picsum returned status {res.status_code}.")
            return None
        content_type = res.headers.get("Content-Type", "").lower()
        if "image/" not in content_type:
            print(f"⚠️ Picsum returned non-image content: {content_type}.")
            return None
        img_ext = _detect_image_extension(content_type)
        img_filename = f"{img_basename}-picsum.{img_ext}"
        img_path = os.path.join(img_dir, img_filename)
        _write_image(img_path, res.content)
        return img_filename
    except Exception as e:
        print(f"⚠️ Picsum fallback failed: {e}")
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

    if _try_gemini_imagen(prompt, img_path):
        img_filename = os.path.basename(img_path)

    if not img_filename:
        img_filename = _try_pollinations(state["topic"], img_basename, img_dir, timestamp)

    if not img_filename:
        print("⚠️ Pollinations failed. Falling back to Picsum placeholder image.")
        img_filename = _try_picsum(img_basename, img_dir, timestamp)

    if not img_filename:
        print("⚠️ All image sources failed. Publishing without a header image.")

    date_str = datetime.date.today().strftime("%Y-%m-%d")
    date_full = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S +0000")
    safe_slug = re.sub(r"[^a-z0-9-]", "", state["topic"].lower().replace(" ", "-"))[:50]
    # Guard against slug collisions on the rare chance two runs land on a
    # near-identical slug the same day.
    safe_slug = f"{safe_slug}-{timestamp % 10000}"

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
    if img_filename:
        print(f"🖼️  IMAGE SAVED: {os.path.join(img_dir, img_filename)}")
    return {"content": final_markdown, "image_url": img_filename or ""}


# --- 4. GRAPH & RUNTIME ---

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
    # "field" is now chosen internally (rotated by day) inside deep_data_diviner,
    # so we no longer hardcode a single generic field here.
    app.invoke({"field": "AI developments", "full_draft": "", "iteration": 0})
