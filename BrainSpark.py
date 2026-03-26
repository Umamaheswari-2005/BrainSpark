import os
import json
import re
import random
import time
import warnings
import logging
import urllib.parse
from urllib.parse import parse_qs, urlparse, unquote

import streamlit as st
from groq import Groq
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import sympy as sp
from bs4 import BeautifulSoup
import requests

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.CRITICAL)

os.environ["GROQ_API_KEY"] = "your_key"
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

os.environ["SERPAPI_API_KEY"] = "your_key"
os.environ["SERPAPI_API_KEY"] = st.secrets["SERPAPI_API_KEY"]

os.environ["CREWAI_TELEMETRY"]  = "false"
os.environ["OTEL_SDK_DISABLED"] = "true"

GROQ_API_KEY    = os.environ["GROQ_API_KEY"]
SERPAPI_API_KEY = os.environ["SERPAPI_API_KEY"]
#SERPAPI_API_KEY = "5uj4paRxJmMLJ2Y4Lao5bN6p"
groq_client     = Groq(api_key=GROQ_API_KEY)

for pkg in ["punkt", "stopwords", "punkt_tab"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

MATH_KEYWORDS = {
    "math","maths","mathematics","algebra","geometry","calculus",
    "trigonometry","statistics","probability","arithmetic","aptitude",
    "quantitative","numerical","number","theory","linear","differential",
    "integral","equation","matrix","matrices","vector","polynomial",
    "logarithm","permutation","combination","percentage","ratio",
    "proportion","average","mean","median","mode","profit","loss",
    "interest","speed","distance","time","work","mensuration","area",
    "volume","derivative","integration","series","sequence","prime",
}

def is_math_topic(topic):
    try:
        tokens = word_tokenize(topic.lower())
        stops  = set(stopwords.words("english"))
        words  = [t for t in tokens if t not in stops]
    except Exception:
        words = topic.lower().split()
    return bool(set(words) & MATH_KEYWORDS)

def build_prompt(topic, difficulty, is_math):
    seed_hint = f"Seed-{random.randint(10000,99999)}-Time-{int(time.time())}"

    if is_math and difficulty == "Easy":
        diff_rules = (
            "Generate ONLY practical calculation-based questions using simple real-life scenarios. "
            "Every question MUST involve actual numbers and require the student to calculate an answer. "
            "Examples: 'A shopkeeper buys 50 items at Rs.20 each and sells at Rs.25. What is the profit?', "
            "'A train travels 120 km in 2 hours. What is its speed?', "
            "'If 30% of a number is 60, what is the number?'. "
            "STRICTLY FORBIDDEN: Do NOT ask theoretical questions like 'What is percentage?', "
            "'Define profit', 'What is the formula for speed?'. "
            "ALL 20 questions must have numbers and require calculation."
        )
    elif is_math and difficulty == "Medium":
        diff_rules = (
            "Generate calculation-based questions with moderate complexity. "
            "Use multi-step problems with real numbers. "
            "No pure theory questions. Every question must require solving or calculating. "
            "Examples: compound interest, time & work with multiple workers, ratio & proportion problems."
        )
    elif is_math and difficulty == "Hard":
        diff_rules = (
            "Generate complex real-world scenario problems requiring multi-step calculations. "
            "Use business scenarios, data interpretation, or applied math problems. "
            "All questions must have specific numbers and require calculation to solve."
        )
    else:
        diff_rules = {
            "Easy":   "Focus on definitions, basic concepts and foundational knowledge.",
            "Medium": "Test expert-level understanding. Do NOT use real-world scenarios.",
            "Hard":   "Every question MUST be a real-world scenario or practical problem.",
        }[difficulty]

    math_note = ""
    if is_math:
        math_note = """
IMPORTANT - MATH TOPIC EXPLANATION FORMAT:
The "explanation" field MUST be written EXACTLY in this structured format (like a textbook solution):

Formula: [Write the relevant formula here]

Given:
- [variable 1] = [value]
- [variable 2] = [value]

Solution:
[variable] = [formula substitution]
           = [intermediate calculation]
           = [final answer with unit]

Therefore, [restate the answer in a sentence].

Rules:
- NEVER write explanation as a single inline sentence like "Step 1: ... Step 2: ..."
- ALWAYS use line breaks between each step
- ALWAYS show the formula first, then given values, then working line by line
- Show each arithmetic operation on its own line
- Include units in every step (km/h, Rs., %, metres, etc.)
- The format must look like a clean textbook solution, NOT a paragraph
"""

    return f"""You are an expert quiz designer. {seed_hint}
Generate EXACTLY 20 UNIQUE MCQ questions on the topic: "{topic}"
Difficulty: {difficulty} — {diff_rules}
{math_note}
CRITICAL RULES FOR UNIQUENESS:
- Every question must be completely different from each other.
- Do NOT repeat the same concept, scenario, or numbers across questions.
- Vary the question style: some with different values, different scenarios, different sub-topics.
- Never generate the same question twice even if this topic was used before.
- Use a wide variety of sub-topics within "{topic}".

Return ONLY a valid JSON array of 20 objects. No markdown. No extra text.
Each object must follow this schema:
{{
  "id": <1-20>,
  "question": "...",
  "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}},
  "correct_answer": "<A|B|C|D>",
  "explanation": "...",
  "resource_query": "<5-8 word search query for this concept>"
}}
Start with [ and end with ]. Nothing else."""

def generate_quiz(topic, difficulty, is_math):
    temperature = 0.95 if is_math else 0.85
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a world-class educational quiz designer. "
                    "You always return raw JSON arrays only — no markdown, no explanation, nothing else. "
                    "For math topics you include step-by-step solutions with formulas in explanations. "
                    "You NEVER repeat questions. Every quiz must be completely fresh and unique. "
                    f"Random session ID: {random.randint(100000, 999999)}"
                ),
            },
            {"role": "user", "content": build_prompt(topic, difficulty, is_math)},
        ],
        temperature=temperature,
        max_tokens=8000,
    )
    result = response.choices[0].message.content
    match  = re.search(r"\[[\s\S]*\]", result)
    if not match:
        raise ValueError("Could not extract JSON from response. Please try again.")
    return json.loads(match.group(0))[:20]

# ── Resource Fetching ─────────────────────────────────────────────────────────
#
# Architecture:
#   _http_get()                  — shared HTTP with retries + exponential backoff
#   _parse_serpapi_response()    — extract web + YouTube from one SerpAPI blob
#   _parse_ddg_html()            — extract web + YouTube from DuckDuckGo HTML
#   _rank_queries_by_tfidf()     — use TF-IDF to pick best query variant
#   fetch_resources_for_question()— full fallback chain per question
#   fetch_all_resources()        — rate-limited loop over all 20 questions
# ─────────────────────────────────────────────────────────────────────────────

# Junk domains whose results should be skipped
_JUNK_DOMAINS = {"duckduckgo.com", "google.com", "bing.com", "yahoo.com", "ask.com"}

def _http_get(url, params=None, headers=None, timeout=18, retries=3):
    """
    HTTP GET with exponential-backoff retries.
    Waits 1s, 2s, 4s between attempts before giving up.
    Returns requests.Response or raises on final failure.
    """
    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    if headers:
        default_headers.update(headers)

    last_exc = None
    for attempt in range(retries):
        try:
            resp = requests.get(
                url,
                params=params,
                headers=default_headers,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt)   # 1s → 2s → 4s
    raise last_exc


def _is_junk(url):
    """Return True if the URL belongs to a search engine or redirect page."""
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
        return any(j in host for j in _JUNK_DOMAINS)
    except Exception:
        return False


def _parse_serpapi_response(data):
    """
    Extract up to 3 web results + 1 YouTube result from a SerpAPI JSON blob.
    Checks organic_results, inline_videos, and video_results blocks.
    Returns (web_list, yt_dict_or_None).
    """
    web, yt = [], None

    for r in data.get("organic_results", []):
        title   = r.get("title", "").strip()
        link    = r.get("link", "").strip()
        snippet = r.get("snippet", "").strip()[:160]
        if not title or not link or not link.startswith("http") or _is_junk(link):
            continue
        is_yt = "youtube.com/watch" in link or "youtu.be/" in link
        if is_yt and yt is None:
            yt = {"title": f"▶ {title}", "link": link, "snippet": snippet or "Watch on YouTube"}
        elif not is_yt and len(web) < 3:
            web.append({"title": title, "link": link, "snippet": snippet})
        if len(web) >= 3 and yt:
            break

    # Check inline_videos block if no YouTube found yet
    if yt is None:
        for v in data.get("inline_videos", [])[:5]:
            link  = v.get("link", "") or v.get("url", "")
            title = v.get("title", "")
            if ("youtube.com/watch" in link or "youtu.be/" in link) and title:
                yt = {"title": f"▶ {title}", "link": link, "snippet": "Watch on YouTube"}
                break

    # Check video_results block
    if yt is None:
        for v in data.get("video_results", [])[:5]:
            link  = v.get("link", "") or v.get("url", "")
            title = v.get("title", "")
            if isinstance(title, dict):
                title = (title.get("runs") or [{}])[0].get("text", "")
            if ("youtube.com/watch" in link or "youtu.be/" in link) and title:
                yt = {"title": f"▶ {title}", "link": link, "snippet": "Watch on YouTube"}
                break

    return web, yt


def _parse_ddg_html(html_text):
    """
    Parse DuckDuckGo HTML search results page.
    Robust to HTML structure changes — tries multiple selector patterns.
    Returns (web_list, yt_dict_or_None).
    """
    soup = BeautifulSoup(html_text, "html.parser")
    web, yt = [], None

    # Try multiple link selector patterns (DDG changes these occasionally)
    candidates = (
        soup.select("a.result__a")
        or soup.select("a[href*='uddg=']")
        or soup.select(".result__url")
    )

    for a in candidates[:20]:
        href  = a.get("href", "")
        title = a.get_text(strip=True)

        # Decode DDG redirect URLs
        if "uddg=" in href:
            real_url = unquote(parse_qs(urlparse(href).query).get("uddg", [href])[0])
        elif href.startswith("http"):
            real_url = href
        else:
            continue

        if not real_url.startswith("http") or _is_junk(real_url):
            continue

        # Grab snippet from sibling element
        parent = (
            a.find_parent(class_="result__body")
            or a.find_parent(class_="result")
            or a.find_parent()
        )
        snippet_tag = (
            parent.find(class_="result__snippet")
            or parent.find(class_="result__desc")
            if parent else None
        )
        snippet = snippet_tag.get_text(strip=True)[:160] if snippet_tag else ""

        is_yt = "youtube.com/watch" in real_url or "youtu.be/" in real_url
        if is_yt and yt is None:
            yt = {"title": f"▶ {title}", "link": real_url, "snippet": "Watch on YouTube"}
        elif not is_yt and len(web) < 3 and title:
            web.append({"title": title, "link": real_url, "snippet": snippet})

        if len(web) >= 3 and yt:
            break

    return web, yt


def _rank_queries_by_tfidf(query_variants, all_queries):
    """
    Use TF-IDF to score query variants against the full corpus of queries.
    Returns the variant most distinct / information-dense.
    Falls back to first variant if TF-IDF fails.
    """
    if len(query_variants) == 1 or not all_queries:
        return query_variants[0]
    try:
        corpus = all_queries + query_variants
        vec    = TfidfVectorizer(max_features=100, stop_words="english")
        mat    = vec.fit_transform(corpus)
        # Score = sum of TF-IDF weights for each variant
        variant_scores = mat[-len(query_variants):].toarray().sum(axis=1)
        best_idx = int(variant_scores.argmax())
        return query_variants[best_idx]
    except Exception:
        return query_variants[0]


def _serpapi_fetch(query):
    """
    One SerpAPI Google call fetching 10 results.
    Returns (web_list, yt_or_None) or raises on failure.
    """
    resp = _http_get(
        "https://serpapi.com/search",
        params={
            "engine":  "google",
            "q":       query,
            "api_key": SERPAPI_API_KEY,
            "num":     10,
            "hl":      "en",
            "gl":      "us",
        },
        timeout=20,
        retries=3,
    )
    data = resp.json()
    if "error" in data:
        raise ValueError(f"SerpAPI: {data['error']}")
    return _parse_serpapi_response(data)


def _ddg_fetch(query):
    """
    DuckDuckGo HTML scrape with retries.
    Returns (web_list, yt_or_None).
    """
    resp = _http_get(
        "https://html.duckduckgo.com/html/",
        params={"q": query},
        timeout=15,
        retries=3,
    )
    return _parse_ddg_html(resp.text)


def fetch_resources_for_question(query, all_queries=None):
    """
    Full fallback chain for one question:
      1. Build 2 query variants, pick best via TF-IDF
      2. Try SerpAPI  (retries built-in)
      3. Try DuckDuckGo  (retries built-in)
      4. Try DuckDuckGo with alternate query variant
    Returns list of up to 4 items: [web1, web2, web3, youtube]
    """
    all_queries = all_queries or []

    # Build query variants — TF-IDF picks the most information-dense one
    variants = [
        f"{query} tutorial learn",
        f"{query} explained guide",
    ]
    best_query = _rank_queries_by_tfidf(variants, all_queries)

    web, yt = [], None

    # ── Layer 1: SerpAPI ─────────────────────────────────────────────────────
    if SERPAPI_API_KEY:
        try:
            web, yt = _serpapi_fetch(best_query)
        except Exception:
            web, yt = [], None

    # ── Layer 2: DuckDuckGo with best query ──────────────────────────────────
    if not web:
        try:
            web, yt_ddg = _ddg_fetch(best_query)
            if yt is None:
                yt = yt_ddg
        except Exception:
            pass

    # ── Layer 3: DuckDuckGo with alternate query variant ─────────────────────
    if not web:
        alt_query = variants[1] if best_query == variants[0] else variants[0]
        try:
            time.sleep(1)   # small pause before retry with alt query
            web, yt_alt = _ddg_fetch(alt_query)
            if yt is None:
                yt = yt_alt
        except Exception:
            pass

    # ── Assemble: web results first, YouTube last ─────────────────────────────
    final = web[:3]
    if yt:
        final.append(yt)
    return final


def fetch_all_resources(questions, topic):
    """
    Fetch 3 web links + 1 YouTube for ALL questions with rate limiting.

    Improvements over naive loop:
    • Rate limit: 1.2s gap between requests (avoids SerpAPI/DDG blocks)
    • TF-IDF: all queries passed as corpus so each question picks
      its most distinctive query variant
    • Per-question retry handled inside fetch_resources_for_question()
    • Never skips a question — failed questions get empty list, not crash
    """
    queries = [q.get("resource_query", topic) for q in questions]

    resources  = {}
    last_fetch = 0.0   # timestamp of last HTTP request

    for i, query in enumerate(queries):
        # Rate limit: enforce minimum 1.2s between fetches
        elapsed = time.time() - last_fetch
        if elapsed < 1.2:
            time.sleep(1.2 - elapsed)

        try:
            resources[i] = fetch_resources_for_question(query, all_queries=queries)
        except Exception:
            resources[i] = []

        last_fetch = time.time()

    return resources

# ── Score Calculator ──────────────────────────────────────────────────────────
def calculate_score(questions, user_answers):
    correct = 0
    results = []
    for i, q in enumerate(questions):
        ua = user_answers.get(i)
        ca = q["correct_answer"]
        ok = ua == ca
        if ok:
            correct += 1
        results.append({
            "question":       q["question"],
            "options":        q.get("options", {}),
            "user_answer":    ua,
            "correct_answer": ca,
            "is_correct":     ok,
            "explanation":    q.get("explanation", ""),
            "resource_query": q.get("resource_query", ""),
        })
    pct   = (correct / len(questions)) * 100
    grade = ("A+" if pct >= 90 else "A" if pct >= 80 else "B" if pct >= 70
             else "C" if pct >= 60 else "D" if pct >= 50 else "F")
    return {"correct": correct, "total": len(questions), "percentage": pct, "grade": grade, "results": results}


# ── Streamlit App ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="BrainSpark ⚡", page_icon="⚡", layout="centered")

st.title("⚡ BrainSpark — AI Quiz Generator")
st.markdown("Powered by **Groq LLaMA-3.3-70B** · DuckDuckGo · NLTK")
st.divider()

for key, default in {
    "stage": "home", "questions": [], "topic": "", "difficulty": "Medium",
    "score_data": {}, "resources": {}, "user_answers": {}
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ════════════════════════════════════════
# HOME STAGE
# ════════════════════════════════════════
if st.session_state.stage == "home":

    topic      = st.text_input("📚 Enter Quiz Topic", placeholder="e.g. Python, Linked List, Algebra, Machine Learning...")
    difficulty = st.radio("🎯 Select Difficulty", ["Easy", "Medium", "Hard"], index=1, horizontal=True)

    is_math_detected = topic and is_math_topic(topic)

    if is_math_detected:
        hints = {
            "Easy":   "🟢 Practical calculation questions with simple real-life scenarios. No theory questions.",
            "Medium": "🟡 Multi-step calculation problems with real numbers.",
            "Hard":   "🔴 Complex real-world math scenarios requiring multi-step solutions.",
        }
    else:
        hints = {
            "Easy":   "🟢 Basic concepts and definitions.",
            "Medium": "🟡 Expert-level knowledge. No real-world scenarios.",
            "Hard":   "🔴 Real-world scenarios and practical problems only.",
        }
    st.caption(hints[difficulty])

    if is_math_detected:
        st.info("🧮 Math/Aptitude topic detected — all questions will be practical calculation-based problems with step-by-step solutions.")

    if st.button("⚡ Generate Quiz", disabled=not bool(topic.strip())):
        with st.spinner("Generating 20 questions... please wait 30–60 seconds."):
            try:
                questions = generate_quiz(topic.strip(), difficulty, is_math_topic(topic.strip()))
                if len(questions) < 5:
                    st.error("Not enough questions generated. Please try a different topic.")
                else:
                    st.session_state.questions    = questions[:20]
                    st.session_state.topic        = topic.strip()
                    st.session_state.difficulty   = difficulty
                    st.session_state.user_answers = {}
                    st.session_state.stage        = "quiz"
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)[:300]}")

# ════════════════════════════════════════
# QUIZ STAGE
# ════════════════════════════════════════
elif st.session_state.stage == "quiz":

    st.subheader(f"📘 Quiz — {st.session_state.topic}")
    st.caption(f"Difficulty: {st.session_state.difficulty}  ·  20 questions  ·  Select one answer per question")
    st.divider()

    questions    = st.session_state.questions
    user_answers = {}

    for i, q in enumerate(questions):
        options = q.get("options", {})
        st.markdown(f"**Q{i+1}. {q['question']}**")
        ans = st.radio(
            f"Answer for Q{i+1}",
            list(options.keys()),
            format_func=lambda x, o=options: f"{x}:  {o[x]}",
            index=None,
            label_visibility="collapsed",
            key=f"q_{i}",
        )
        if ans:
            user_answers[i] = ans
        st.divider()

    answered = len(user_answers)
    if answered < len(questions):
        st.warning(f"⚠️  {len(questions) - answered} question(s) unanswered. Please answer all questions to submit.")

    if st.button("📊 Submit Quiz", disabled=answered < len(questions)):
        score_data = calculate_score(questions, user_answers)
        with st.spinner("Fetching learning resources for all 20 questions..."):
            try:
                resources = fetch_all_resources(questions, st.session_state.topic)
            except Exception:
                resources = {}
        st.session_state.score_data   = score_data
        st.session_state.resources    = resources
        st.session_state.user_answers = user_answers
        st.session_state.stage        = "results"
        st.rerun()

# ════════════════════════════════════════
# RESULTS STAGE
# ════════════════════════════════════════
elif st.session_state.stage == "results":

    score_data = st.session_state.score_data
    resources  = st.session_state.resources or {}
    pct        = score_data["percentage"]
    grade      = score_data["grade"]
    correct    = score_data["correct"]
    total      = score_data["total"]

    perf = ("Outstanding! 🎉" if pct >= 90 else "Great Job! 👏" if pct >= 80
            else "Well Done! 👍" if pct >= 70 else "Keep Going! 💪" if pct >= 50
            else "Keep Practicing! 📚")

    st.subheader("📊 Your Results")
    st.metric("Score", f"{pct:.1f}%", perf)

    c1, c2, c3 = st.columns(3)
    c1.metric("✅ Correct",   correct)
    c2.metric("❌ Incorrect", total - correct)
    c3.metric("🏆 Grade",    grade)
    st.divider()

    st.subheader("📝 Detailed Review")

    for i, r in enumerate(score_data["results"]):
        icon = "✅" if r["is_correct"] else "❌"
        with st.expander(f"{icon}  Q{i+1}: {r['question'][:80]}{'...' if len(r['question']) > 80 else ''}"):

            st.markdown(f"**{r['question']}**")
            st.write("")

            for k, v in r.get("options", {}).items():
                is_user    = k == r["user_answer"]
                is_correct = k == r["correct_answer"]
                if is_correct and is_user:
                    st.success(f"{k}:  {v}   ✓  Your Answer (Correct!)")
                elif is_correct:
                    st.success(f"{k}:  {v}   ✓  Correct Answer")
                elif is_user:
                    st.error(f"{k}:  {v}   ✗  Your Answer")
                else:
                    st.write(f"{k}:  {v}")

            st.divider()
            st.markdown("**📖 Explanation**")
            explanation = r.get("explanation", "No explanation available.")
            is_math_expl = any(kw in explanation for kw in ["Formula:", "Given:", "Solution:", "Therefore,", "Step 1", "="])
            if is_math_expl:
                st.code(explanation, language=None)
            else:
                st.info(explanation)

            st.divider()
            st.markdown("**🔗 Learning Resources**")
            res_list = resources.get(i, [])
            if res_list:
                for res in res_list[:4]:
                    is_yt  = "youtube.com" in res.get("link", "") or "youtu.be" in res.get("link", "")
                    icon   = "▶️" if is_yt else "🔗"
                    st.markdown(f"{icon} **[{res['title']}]({res['link']})**")
                    st.caption(res["link"])
                    if res.get("snippet"):
                        st.caption(res["snippet"][:150])
                    st.write("")

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Retake Quiz", use_container_width=True):
            st.session_state.user_answers = {}
            st.session_state.stage = "quiz"
            st.rerun()
    with col_b:
        if st.button("🏠 New Quiz", use_container_width=True):
            st.session_state.stage        = "home"
            st.session_state.questions    = []
            st.session_state.topic        = ""
            st.session_state.difficulty   = "Medium"
            st.session_state.score_data   = {}
            st.session_state.resources    = {}
            st.session_state.user_answers = {}
            st.rerun()
