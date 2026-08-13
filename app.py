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
import chromadb
import open_clip
import torch
from PIL import Image
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

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
# via session state (see note in run_search) since tool-calling LLMs
# work with text/JSON arguments, not raw image bytes, in their
# function-calling schema.
# ------------------------------------------------------------------

@tool
def search_similar_sarees(top_k: int = 5) -> str:
    """Search the saree catalogue for images visually similar to the
    image the user just uploaded. Call this whenever the user has
    uploaded/attached an image and is asking to find similar, matching,
    or related sarees. Returns a JSON string of the top matches with
    their name, similarity score, and product link.

    Args:
        top_k: number of similar results to return (default 5, max 10).
    """
    top_k = min(max(top_k, 1), 10)
    query_embedding = st.session_state.get("pending_query_embedding")
    if query_embedding is None:
        return json.dumps({"error": "No uploaded image found to search with."})

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
    return json.dumps(matches)


# ------------------------------------------------------------------
# Agent wiring
# ------------------------------------------------------------------

def build_agent():
    # Requires GOOGLE_API_KEY in the environment (free tier from
    # aistudio.google.com — set via Streamlit secrets when deployed).
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are TailorTalk's shopping assistant for a saree catalogue. "
         "When a user has uploaded an image and asks to find similar, "
         "matching, or related items, call the search_similar_sarees tool. "
         "After getting results, describe them naturally and briefly — "
         "mention colour/fabric similarity, not just that they're 'sarees'. "
         "If no image has been uploaded yet, ask the user to upload one."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    tools = [search_similar_sarees]
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)


# ------------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------------

st.title("🧵 TailorTalk — Saree Finder")
st.caption("Upload a saree photo and chat naturally to find visually similar pieces.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()

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
            chat_history = []
            for m in st.session_state.messages[:-1]:
                role = "human" if m["role"] == "user" else "ai"
                chat_history.append((role, m["content"]))

            result = st.session_state.agent.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
            response_text = result["output"]
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
