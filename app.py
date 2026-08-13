"""
TailorTalk — Visual Saree Similarity Agent
============================================
A chat agent that lets a user upload a saree image and get back visually
similar products from the catalogue, using CLIP embeddings + ChromaDB for
search, wrapped as a LangChain tool the LLM calls when appropriate.

Run locally with:  streamlit run app.py
Deploy on Streamlit Community Cloud by pushing this repo to GitHub and
connecting it there (see README.md).
"""

import os
import io
import json

import streamlit as st

# Streamlit's secrets manager does NOT automatically become a regular
# environment variable inside the app — it has to be wired through
# explicitly, or libraries that read os.environ (like the Google SDK
# underneath langchain_google_genai) won't see it at all.
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

import chromadb
import open_clip
import torch
from PIL import Image
import google.generativeai as genai

# ------------------------------------------------------------------
# Setup — loaded once per session, not per message
# ------------------------------------------------------------------

st.set_page_config(page_title="TailorTalk — Saree Finder", page_icon="🧵")

@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    return model.to(device).eval(), preprocess, device

@st.cache_resource
def load_chroma():
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_collection("sarees")

model, preprocess, device = load_clip()
collection = load_chroma()


def get_center_crop(img: Image.Image, crop_ratio: float = 0.6) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * crop_ratio), int(h * crop_ratio)
    left, top = (w - cw) // 2, (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch))


@torch.no_grad()
def embed_image(img: Image.Image):
    full_t = preprocess(img).unsqueeze(0).to(device)
    crop_t = preprocess(get_center_crop(img)).unsqueeze(0).to(device)
    full_e = model.encode_image(full_t)
    crop_e = model.encode_image(crop_t)
    full_e = full_e / full_e.norm(dim=-1, keepdim=True)
    crop_e = crop_e / crop_e.norm(dim=-1, keepdim=True)
    combined = (0.5 * full_e + 0.5 * crop_e)
    combined = combined / combined.norm(dim=-1, keepdim=True)
    return combined.cpu().numpy()[0]


# ------------------------------------------------------------------
# The agent tool — this is what the LLM calls when it decides a
# similarity search is being requested. The image itself is passed
# via session state since tool-calling LLMs work with text/JSON
# arguments, not raw image bytes, in their function-calling schema.
# ------------------------------------------------------------------

def search_similar_sarees(top_k: int = 5) -> dict:
    """Search the saree catalogue for images visually similar to the
    image the user just uploaded. Call this whenever the user has
    uploaded/attached an image and is asking to find similar, matching,
    or related sarees. Returns the top matches with their name,
    similarity score, and product link.

    Args:
        top_k: number of similar results to return (default 5, max 10).
    """
    top_k = min(max(int(top_k), 1), 10)
    query_embedding = st.session_state.get("pending_query_embedding")
    if query_embedding is None:
        return {"error": "No uploaded image found to search with."}

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=top_k,
    )

    matches = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        # Cosine distance -> similarity score (1 = identical, 0 = unrelated)
        similarity = round(1 - distance, 4)
        matches.append({
            "name": meta["name"],
            "similarity": similarity,
            "price": meta["price"],
            "link": meta["website_link"],
            "image_url": meta["image_url"],
        })

    st.session_state["last_matches"] = matches
    return {"matches": matches}


TOOL_FUNCTIONS = {"search_similar_sarees": search_similar_sarees}

SYSTEM_INSTRUCTION = (
    "You are TailorTalk's shopping assistant for a saree catalogue. "
    "When a user has uploaded an image and asks to find similar, "
    "matching, or related items, call the search_similar_sarees tool. "
    "After getting results, describe them naturally and briefly — "
    "mention colour/fabric similarity, not just that they're 'sarees'. "
    "If no image has been uploaded yet, ask the user to upload one."
)


# ------------------------------------------------------------------
# Agent wiring — calls Google's SDK directly rather than through the
# langchain_google_genai wrapper, which has a known bug forcing a
# stale API path that 404s regardless of model name or version.
# ------------------------------------------------------------------

@st.cache_resource
def build_agent():
    api_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
    if not api_key:
        st.error(
            "GOOGLE_API_KEY is missing or empty. Check Streamlit Cloud → "
            "your app → Settings → Secrets, and confirm it's saved exactly as:\n\n"
            'GOOGLE_API_KEY = "your-key-here"'
        )
        st.stop()
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        tools=[search_similar_sarees],
        system_instruction=SYSTEM_INSTRUCTION,
    )
    return model


def run_agent_turn(chat_session, user_input: str):
    """Sends a message, executes any tool call the model requests, and
    returns the final natural-language response."""
    response = chat_session.send_message(user_input)

    for part in response.parts:
        fn = getattr(part, "function_call", None)
        if fn and fn.name in TOOL_FUNCTIONS:
            args = dict(fn.args) if fn.args else {}
            result = TOOL_FUNCTIONS[fn.name](**args)
            response = chat_session.send_message(
                genai.protos.Content(
                    parts=[genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=fn.name,
                            response={"result": result},
                        )
                    )]
                )
            )
    return response.text


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.title("🧵 TailorTalk — Saree Finder")
st.caption("Upload a saree photo and chat naturally to find visually similar pieces.")

# Temporary diagnostic — confirms the secret is actually reaching the app
# without exposing the key itself. Safe to remove once things work.
_debug_key = st.secrets.get("GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY"))
with st.sidebar:
    if _debug_key:
        st.caption(f"✅ API key loaded ({len(_debug_key)} chars, starts with '{_debug_key[:4]}...')")
    else:
        st.caption("❌ No API key found in secrets or environment")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "gemini_model" not in st.session_state:
    st.session_state.gemini_model = build_agent()
if "chat_session" not in st.session_state:
    st.session_state.chat_session = st.session_state.gemini_model.start_chat()

uploaded_file = st.file_uploader("Upload a saree image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    query_img = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("RGB")
    st.image(query_img, caption="Your uploaded image", width=200)
    st.session_state["pending_query_embedding"] = embed_image(query_img)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("matches"):
            cols = st.columns(min(len(msg["matches"]), 5))
            for col, m in zip(cols, msg["matches"]):
                with col:
                    st.image(m["image_url"], use_container_width=True)
                    st.caption(f"{m['name'][:30]}…\nSimilarity: {m['similarity']}")
                    st.markdown(f"[View]({m['link']})")

if user_input := st.chat_input("Ask me to find similar sarees…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            response_text = run_agent_turn(st.session_state.chat_session, user_input)
            matches = st.session_state.pop("last_matches", None)

            st.write(response_text)
            if matches:
                cols = st.columns(min(len(matches), 5))
                for col, m in zip(cols, matches):
                    with col:
                        st.image(m["image_url"], use_container_width=True)
                        st.caption(f"{m['name'][:30]}…\nSimilarity: {m['similarity']}")
                        st.markdown(f"[View]({m['link']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_text,
        "matches": matches,
    })
