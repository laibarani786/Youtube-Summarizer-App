"""
YouTube Summarization - Enhanced Summarizer (Final)
Save as: YouTube_Summarization.py
Run with:
    streamlit run YouTube_Summarization.py
"""
import os, re, json, tempfile, io, time
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import requests
import validators
import streamlit as st
import streamlit.components.v1 as components

# LLM & Langchain
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader, PyPDFLoader
from langchain_groq import ChatGroq

# YouTube helpers
from yt_dlp import YoutubeDL
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled, VideoUnavailable

# TTS
from gtts import gTTS

# small utils
from collections import Counter
from datetime import datetime

# ---------- Config / Constants ----------
HISTORY_FILE = Path("summary_history.json")
ASSETS_DIR = Path("assets")
ARCH_IMG = ASSETS_DIR / "architecture.png"

STOPWORDS = set([
    "the","and","is","in","to","of","a","for","on","that","this","it","with","as","are","be","by","an","from","at"
])

# ---------- Page setup ----------
st.set_page_config(page_title="YouTube Summarization - Enhanced", page_icon="🧠", layout="wide")
st.title("🧠 YouTube Summarization - Enhanced Summarizer")
st.caption("Streamlit + Langchain + Groq — advanced features added")

# show architecture diagram if present
if ARCH_IMG.exists():
    st.markdown("---")
    st.write("### 🏗 Architecture Diagram")
    st.image(str(ARCH_IMG), use_column_width=True, caption="YouTube Summarization Architecture")
    with open(ARCH_IMG, "rb") as f:
        st.download_button("⬇ Download Architecture Diagram", data=f, file_name="architecture.png", mime="image/png")
    st.markdown("---")

# ---------- Sidebar ----------
with st.sidebar:
    st.subheader("🔑 API & Model")
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY",""))[:100]
    model = st.selectbox("Groq Model", ["llama-3.1-8b-instant","gemma2-9b-it","llama-3.1-70b-versatile"], index=0)
    custom_model = st.text_input("Custom Model (optional)")

    st.subheader("🧠 Generation")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    out_len = st.slider("Target summary length (words)", 90, 800, 300, 20)

    st.subheader("✍ Style")
    out_style = st.selectbox("Output Style", ["Bullets","Paragraph","Both"])
    tone = st.selectbox("Tone", ["Neutral","Formal","Casual","Executive Brief"])
    out_lang = st.selectbox("Language", ["English","Urdu","Roman Urdu","Auto"])

    st.subheader("⚙ Processing")
    chain_mode = st.radio("Chain Type", ["Auto","Stuff","Map-Reduce"], index=0)
    chunk_size = st.slider("Chunk Size (characters)", 500, 4000, 1600, 100)
    chunk_overlap = st.slider("Chunk Overlap (characters)", 0, 800, 150, 10)
    max_map_chunks = st.slider("Max chunks (combine)", 9, 64, 28, 1)

    st.subheader("Extras")
    show_preview = st.checkbox("Show source preview", value=True)
    want_outline = st.checkbox("Produce outline", value=True)
    want_keywords = st.checkbox("Extract keywords & hashtags", value=True)

    st.markdown("---")
    st.write("*Saved History*")
    if HISTORY_FILE.exists():
        try:
            hdata = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            for i, ent in enumerate(hdata[:10]):
                st.write(f"{i+1}. {ent.get('title', 'Untitled')} — {ent.get('ts','')}")
        except Exception:
            st.write("No history yet.")
    st.markdown("---")
    st.write("Made with ❤ — YouTube Summarization")

# ---------- Main Input ----------
left, right = st.columns([2,1])
with left:
    url = st.text_input("Paste URL (website, YouTube, or direct PDF link)")
with right:
    uploaded = st.file_uploader("...or upload a PDF", type=["pdf"])

# ---------- Helper functions ----------
def is_youtube(u: str) -> bool:
    try:
        netloc = urlparse(u).netloc.lower()
        return any(host in netloc for host in ["youtube.com","youtu.be"])
    except Exception:
        return False

def normalize_youtube_url(u: str) -> str:
    """Fix weird extra params like `?si=...` that sometimes break loaders."""
    try:
        p = urlparse(u)
        if "youtu.be" in p.netloc:
            # keep youtu.be short link
            return u.split("?")[0]
        if "youtube.com" in p.netloc:
            qs = parse_qs(p.query)
            v = qs.get("v", [None])[0]
            if v:
                return f"https://www.youtube.com/watch?v={v}"
            # sometimes mobile URLs or watch?v=...
            return u.split("&si=")[0]
    except Exception:
        pass
    return u

def head_content_type(u: str, timeout=12) -> str | None:
    try:
        r = requests.head(u, allow_redirects=True, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        return (r.headers.get("Content-Type") or "").lower()
    except Exception:
        return None

def clean_caption_text(text: str) -> str:
    text = re.sub(r"\[(?:music|applause|laughter| .*?)]"," ", text, flags=re.I)
    text = re.sub(r"\s+"," ", text)
    return text.strip()

def json3_to_text(s: str) -> str:
    try:
        data = json.loads(s)
        lines=[]
        for ev in data.get("events",[]):
            for seg in ev.get("segs",[]) or []:
                t = seg.get("utf8","")
                if t:
                    lines.append(t.replace("\n"," "))
        return clean_caption_text(" ".join(lines))
    except Exception:
        return clean_caption_text(s)

def fetch_caption_text(cap_url: str) -> str:
    resp = requests.get(cap_url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
    ctype = (resp.headers.get("Content-Type") or "").lower()
    body = resp.text
    if "text/vtt" in ctype or cap_url.endswith(".vtt"):
        out=[]
        for line in body.splitlines():
            s=line.strip()
            if ("-->" in s) or s.isdigit() or s.upper().startswith("WEBVTT"):
                continue
            if s:
                out.append(s)
        return clean_caption_text(" ".join(out))
    if "application/json" in ctype or cap_url.endswith(".json3") or body.strip().startswith("{"):
        return json3_to_text(body)
    return clean_caption_text(body)

def build_llm(groq_api_key: str, model: str, temperature: float):
    chosen = (custom_model.strip() if custom_model else model)
    # defensive init
    try:
        return ChatGroq(groq_api_key=groq_api_key, model_name=chosen, temperature=temperature)
    except TypeError:
        pass
    except Exception:
        raise
    try:
        return ChatGroq(groq_api_key=groq_api_key, model=chosen, temperature=temperature)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize ChatGroq with model '{chosen}': {e}")

def build_prompts(out_len: int, out_style: str, tone: str, want_outline: bool, want_keywords: bool, out_lang: str):
    map_template = """
    Summarize the following text in 3–6 crisp bullet points, maximum 80 words total.
    Keep only the core facts/claims.

    TEXT:
    {text}
    """
    map_prompt = PromptTemplate(template=map_template, input_variables=["text"])

    style_map = {
        "Bullets": "Return crisp bullet points only",
        "Paragraph": "Return one cohesive paragraph only",
        "Both": "Start with 6–10 concise bullet points, then a cohesive paragraph",
    }
    tone_map = {
        "Neutral":"neutral, information-dense",
        "Formal":"formal and precise",
        "Casual":"casual and friendly",
        "Executive Brief":"executive, top-down, action-oriented",
    }
    lang = "Match the user's language." if out_lang=="Auto" else f"Write in {out_lang}."
    extras=[]
    if want_outline:
        extras.append("Provide a short outline with top 3–6 sections.")
    if want_keywords:
        extras.append("Extract 8–12 keywords and 5–8 suggested hashtags.")
    extras_text = ("\n- " + "\n- ".join(extras)) if extras else ""
    combine_template = f"""
    You will receive multiple mini-summaries of different parts of the same source.
    Combine them into a single, faithful summary.

    Constraints and style:
    - Target length = {out_len} words.
    - Output Style: {style_map[out_style]}
    - Tone: {tone_map[tone]}
    - {lang}
    - Be faithful to the source; do not invent facts.
    - If the content is opinionated, label opinions as opinions.
    - Avoid repetition.
    {extras_text}

    Return only the summary (and requested sections); no preambles.

    INPUT_SUMMARIES:
    {{text}}
    """
    combine_prompt = PromptTemplate(template=combine_template, input_variables=["text"])
    return map_prompt, combine_prompt

def choose_chain_type(chain_mode: str, docs: list) -> str:
    if chain_mode != "Auto":
        return chain_mode.lower().replace("-", "_")
    total_chars = sum(len(d.page_content or "") for d in docs)
    return "map_reduce" if total_chars > 15000 else "stuff"

def even_sample(docs, k:int):
    n=len(docs)
    if k>=n:
        return docs
    idxs=[round(i*(n-1)/(k-1)) for i in range(k)]
    return [docs[i] for i in idxs]

def load_youtube_docs(url: str):
    """Try multiple strategies to extract a useful text for YouTube video:
       1) YoutubeLoader (langchain)
       2) youtube_transcript_api
       3) yt-dlp subtitles / automatic captions
       4) fallback to video description (so there is at least some text)
    """
    url = normalize_youtube_url(url)
    # 1) try YoutubeLoader
    try:
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True,
                                               language=["en","en-US","en-GB","ur","hi"], translation=None)
        docs = loader.load()
        if docs and any((d.page_content or "").strip() for d in docs):
            # try transcript api too to get timestamps
            try:
                parsed = urlparse(url)
                vid = None
                if "youtu.be" in parsed.netloc:
                    vid = parsed.path.lstrip("/")
                else:
                    vid = parse_qs(parsed.query).get("v", [None])[0]
                if vid:
                    try:
                        transcript = YouTubeTranscriptApi.get_transcript(vid, languages=['en','ur','hi'])
                        return docs, {"type":"youtube", "timestamps": transcript}
                    except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
                        return docs, {"type":"youtube"}
            except Exception:
                return docs, {"type":"youtube"}
    except Exception:
        pass

    # 2/3) fallback via yt-dlp
    ydl_opts = {"skip_download": True, "quiet": True}
    with YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except Exception as e:
            raise RuntimeError(f"yt-dlp failed to fetch video info: {e}")
        caps = info.get("subtitles") or {}
        auto_caps = info.get("automatic_captions") or {}
        # try to find a caption URL
        def first_track_url(track_dict, langs=("en","en-US","en-GB","ur","hi")):
            for lg in langs:
                tracks = track_dict.get(lg) or []
                if tracks:
                    url0 = tracks[0].get("url")
                    if url0:
                        return url0
            return None

        cap_url = first_track_url(caps) or first_track_url(auto_caps)
        if cap_url:
            text = fetch_caption_text(cap_url)
            from langchain.schema import Document
            return [Document(page_content=text, metadata={"source": url})], {"type":"youtube_fallback"}

        # 4) try video description as fallback (better than nothing)
        description = info.get("description") or ""
        title = info.get("title") or ""
        if description.strip():
            from langchain.schema import Document
            return [Document(page_content=f"{title}\n\n{description}", metadata={"source": url})], {"type":"youtube_description"}
        # if nothing usable
        raise RuntimeError("No captions or description available for this video. Video may have no transcript (e.g., music-only).")

@st.cache_data(show_spinner=False)
def fetch_and_load(url: str, chunk_size:int, chunk_overlap:int):
    meta = {"source": url, "type":"html", "title": None}
    if is_youtube(url):
        docs, yt_meta = load_youtube_docs(url)
        meta.update(yt_meta)
        try:
            if docs and docs[0].metadata.get("title"):
                meta["title"] = docs[0].metadata["title"]
        except Exception:
            pass
        return docs, meta

    ctype = head_content_type(url) or ""
    if "pdf" in ctype or url.lower().endswith(".pdf"):
        meta["type"] = "pdf"
        with requests.get(url, stream=True, timeout=20, headers={"User-Agent":"Mozilla/5.0"}) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        tmp.write(chunk)
                tmp_path = tmp.name
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        return docs, meta

    # webpage
    try:
        loader = WebBaseLoader([url])
        docs = loader.load()
        if docs and docs[0].metadata.get("title"):
            meta["title"] = docs[0].metadata["title"]
    except Exception:
        html = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"}).text
        # handle Google "Sorry..." page or similar
        if "<title>Sorry" in html or "automated queries" in html.lower():
            raise RuntimeError("Remote host blocked automated requests (site returned anti-bot page). Try a different public URL.")
        from langchain.schema import Document
        text = re.sub(r"<[^>]+>", " ", html)
        docs = [Document(page_content=text, metadata={"source":url})]

    # splitting into chunks if large
    if docs and sum(len(d.page_content or "") for d in docs) > chunk_size * 1.5:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                                  separators=["\n\n","\n",".","?","!", " "])
        out=[]
        for d in docs:
            out.extend(splitter.split_documents([d]))
        return out, meta
    return docs, meta

def load_pdf_from_upload(uploaded_file, chunk_size:int, chunk_overlap:int):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()
    if docs and sum(len(d.page_content or "") for d in docs) > chunk_size * 1.5:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        parts=[]
        for d in docs:
            parts.extend(splitter.split_documents([d]))
        return parts
    return docs

def run_chain(llm, docs, map_prompt: PromptTemplate, combine_prompt: PromptTemplate, mode: str, max_map_chunks:int) -> str:
    mode = mode.lower().replace("-", "_")
    if mode == "stuff":
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=combine_prompt)
    else:
        if len(docs) > max_map_chunks:
            docs = even_sample(docs, max_map_chunks)
            st.info(f"Long source: sampled {max_map_chunks} chunks evenly to fit the context.")
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt)
    try:
        try:
            res = chain.invoke({"input_documents": docs})
            return res["output_text"] if isinstance(res, dict) and "output_text" in res else str(res)
        except Exception:
            return chain.run(input_documents=docs)
    except Exception as e:
        raise

# ---------- New utilities ----------
def extract_keywords(text, top_n=10):
    words = re.findall(r"\w{3,}", (text or "").lower())
    words = [w for w in words if w not in STOPWORDS and not w.isdigit()]
    counts = Counter(words)
    return [w for w,c in counts.most_common(top_n)]

def save_history(entry: dict):
    all_data = []
    if HISTORY_FILE.exists():
        try:
            all_data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            all_data = []
    all_data.insert(0, entry)
    HISTORY_FILE.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")

def build_chapter_summaries_from_timestamps(transcript_list, chunk_seconds=120):
    chapters=[]
    cur_text=[]
    cur_start=None
    cur_end=None
    for seg in transcript_list:
        if cur_start is None:
            cur_start = seg.get('start',0)
            cur_end = seg.get('start',0) + seg.get('duration',0)
        cur_text.append(seg.get('text',''))
        cur_end = seg.get('start',0) + seg.get('duration',0)
        if (cur_end - cur_start) >= chunk_seconds:
            chapters.append({"start":int(cur_start), "end":int(cur_end), "text":" ".join(cur_text)})
            cur_text=[]; cur_start=None; cur_end=None
    if cur_text:
        chapters.append({"start":int(cur_start or 0), "end":int(cur_end or 0), "text":" ".join(cur_text)})
    return chapters

def detect_urdu(text: str) -> bool:
    # very simple heuristic: check for Urdu/Arabic script characters
    return bool(re.search(r'[\u0600-\u06FF]', text or ""))

# ---------- Run UI ----------
st.markdown("### 🚀 Run")
go = st.button("Summarize")

if go:
    if not groq_api_key.strip():
        st.error("Provide your Groq API key in the sidebar")
        st.stop()

    # Extra small controls (placed here so they are visible after run button)
    with st.expander("⚡ Extra Features (run-time)"):
        auto_tts = st.checkbox("Auto-detect language for TTS", value=True)
        download_pdf = st.checkbox("Enable PDF download", value=False)
        chapter_duration = st.slider("Chapter summarization duration (s)", 60, 600, 120, 30)
        send_email = st.text_input("Optional: Enter email to send summary")

    docs, meta = [], {"type": None, "source": None, "title": None}
    stage = ""
    kw = []
    try:
        stage = "loading source"
        with st.spinner("Loading source..."):
            if uploaded is not None:
                docs = load_pdf_from_upload(uploaded, chunk_size, chunk_overlap)
                meta.update({"type":"pdf","source": getattr(uploaded,"name",None)})
            elif url.strip():
                if not validators.url(url):
                    st.error("Please enter a valid URL.")
                    st.stop()
                try:
                    docs, meta = fetch_and_load(url, chunk_size, chunk_overlap)
                except Exception as e:
                    st.error(f"Failed to fetch source: {e}")
                    st.stop()
            else:
                st.error("Provide a URL or upload a PDF.")
                st.stop()

            if not docs or not any((d.page_content or "").strip() for d in docs):
                st.error("Could not extract text from the source.")
                st.stop()

        # preview
        if show_preview:
            with st.expander("🔍 Source preview"):
                preview = "".join(d.page_content or "" for d in docs[:3])[:500].strip()
                st.write(f"*Detected type:* {meta.get('type')}")
                if meta.get("title"): st.write(f"*Title:* {meta.get('title')}")
                st.text_area("Preview (first ~500 chars)", preview, height=180)

        # build LLM + prompts
        stage = "initializing LLM"
        llm = build_llm(groq_api_key, model, temperature)
        stage = "building prompts"
        map_prompt, combine_prompt = build_prompts(out_len, out_style, tone, want_outline, want_keywords, out_lang)
        stage = "selecting chain"
        mode = choose_chain_type(chain_mode, docs)

        # run chain with pseudo progress
        stage = f"running chain ({mode})"
        with st.spinner(f"Summarizing via {(custom_model or model)} ({mode})..."):
            prog = st.progress(0)
            try:
                for p in range(0, 90, 10):
                    prog.progress(p+5)
                    time.sleep(0.02)
                summary = run_chain(llm, docs, map_prompt, combine_prompt, mode, max_map_chunks)
                prog.progress(100)
            finally:
                try: prog.empty()
                except: pass

        st.success("Done.")
        st.subheader("✅ Summary")
        st.write(summary if summary else "No summary returned.")

        # keywords
        if want_keywords:
            kw = extract_keywords(summary, top_n=12)
            if kw:
                st.write("*Keywords:*", ", ".join(kw))

        # copy to clipboard (JS) — safe using json.dumps
        try:
            safe_text_js = json.dumps(summary or "")
            copy_html = f"""
            <button onclick="navigator.clipboard.writeText({safe_text_js})">Copy Summary</button>
            """
            components.html(copy_html, height=50)
        except Exception:
            st.button("Copy (fallback)")

        # downloads: txt, md
        st.download_button("⬇ Download .txt", data=summary or "", file_name="summary.txt", mime="text/plain")
        st.download_button("⬇ Download .md", data=f"# Summary\n\n{summary or ''}\n", file_name="summary.md", mime="text/markdown")
        if download_pdf:
            try:
                from fpdf import FPDF  # optional dependency
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", size=12)
                for line in (summary or "").split("\n"):
                    pdf.multi_cell(0, 6, line)
                tmp = io.BytesIO()
                pdf.output(tmp)
                tmp.seek(0)
                st.download_button("⬇ Download summary as PDF", data=tmp, file_name="summary.pdf", mime="application/pdf")
            except Exception:
                # fpdf likely not installed; skip quietly
                pass

        # TTS (auto-detect if requested)
        try:
            if auto_tts or out_lang == "Auto":
                tts_lang_code = "ur" if detect_urdu(summary or "") else "en"
            else:
                tts_lang_code = "en" if out_lang in ("English","Auto","Roman Urdu") else "ur"
            tts = gTTS(text=(summary or ""), lang=tts_lang_code, slow=False)
            mp3_buf = io.BytesIO()
            tts.write_to_fp(mp3_buf)
            mp3_buf.seek(0)
            st.audio(mp3_buf, format="audio/mp3")
            st.download_button("⬇ Download summary as MP3", data=mp3_buf, file_name="summary.mp3", mime="audio/mp3")
        except Exception as e:
            st.warning("TTS generation failed: " + str(e))

        # Chapters (if youtube timestamps exist) — use chosen chapter_duration
        if meta.get("timestamps"):
            st.markdown("### ⏱ Chapter Highlights (approx)")
            chapters = build_chapter_summaries_from_timestamps(meta["timestamps"], chunk_seconds=chapter_duration)
            for i,ch in enumerate(chapters[:10]):
                st.write(f"*Part {i+1} — {ch['start']}s to {ch['end']}s*")
                st.write(ch['text'][:400] + ("..." if len(ch['text'])>400 else ""))
                if st.button(f"Summarize part {i+1}", key=f"summ_part_{i}"):
                    mini_docs = [{"page_content": ch['text']}]
                    with st.spinner("Summarizing chapter..."):
                        try:
                            mini = run_chain(llm, mini_docs, map_prompt, combine_prompt, "stuff", max_map_chunks)
                            st.write(mini)
                        except Exception as e:
                            st.warning("Chapter summarization failed: "+str(e))

        # Q&A block (uses summary as context)
        with st.expander("💬 Ask questions about this content"):
            user_q = st.text_input("Ask a question")
            if st.button("Get Answer", key="get_answer"):
                if not user_q.strip():
                    st.warning("Type a question first.")
                else:
                    prompt = f"Use the following content to answer the question faithfully. CONTENT:\n\n{summary}\n\nQUESTION: {user_q}\nAnswer concisely and label uncertainty if unsure."
                    try:
                        answer = None
                        try:
                            resp = llm.generate([{"role":"user","content":prompt}])
                            if isinstance(resp, (list,tuple)) and len(resp)>0:
                                cand = resp[0]
                                if hasattr(cand, 'content'):
                                    answer = cand.content
                                elif isinstance(cand, dict) and 'content' in cand:
                                    answer = cand['content']
                                else:
                                    answer = str(resp)
                            else:
                                answer = str(resp)
                        except Exception:
                            try:
                                resp2 = llm.chat([{"role":"user","content":prompt}])
                                if isinstance(resp2, (list,tuple)) and len(resp2)>0 and hasattr(resp2[0],'content'):
                                    answer = resp2[0].content
                                else:
                                    answer = str(resp2)
                            except Exception:
                                try:
                                    out = llm(prompt)
                                    answer = out if isinstance(out, str) else str(out)
                                except Exception as e:
                                    answer = f"Failed to get answer from LLM: {e}"

                        st.write("*Answer:*")
                        st.write(answer)
                    except Exception as e:
                        st.error("Q&A failed: "+str(e))

        # Save history
        try:
            kw_final = kw if kw else extract_keywords(summary or "",8)
            entry = {
                "title": meta.get("title") or meta.get("source") or (getattr(uploaded,'name',None) if uploaded else url),
                "source": meta.get("source") or url or (getattr(uploaded,'name',None) if uploaded else None),
                "type": meta.get("type"),
                "summary": summary,
                "keywords": kw_final,
                "ts": datetime.utcnow().isoformat()
            }
            save_history(entry)
            st.info("Saved summary to local history.")
        except Exception as e:
            st.warning("History save failed: " + str(e))

    except Exception as e:
        # final catch: show stage + traceback
        st.error(f"Failed during *{stage}* -> {type(e).__name__}: {e}")
        import traceback; st.code(traceback.format_exc())

# ---------- Notes ----------
with st.expander("🚨 Notes: What works vs. what to avoid"):
    st.markdown("""
- *Best:* Public webpages, YouTube Videos with captions (or auto-caption), direct PDF links, and uploaded PDFs.
- *If a YouTube video has NO captions and NO description (music-only or private)*, summarizer cannot extract text — the script will show an informative error.
- *If a site blocks automated requests (Google anti-bot pages)* the script will surface that error — use a public URL or a different source.
- *History:* summary_history.json is local — for multi-user deploy, use a database.
""")

# EOF
