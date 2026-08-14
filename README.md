# TailorTalk — Visual Saree Similarity Agent

An AI chat agent that finds visually similar sarees from a 1,070-item catalogue,
given an uploaded image. Built for the TailorTalk take-home assignment.

**Live app**: https://tailortalk-sarees.streamlit.app
**Repo**: https://github.com/kumartanishq07-commits/tailortalk-saree-agent

## How it works

1. **Embedding**: Each catalogue image is embedded using OpenAI's CLIP
   (ViT-B/32) via `open_clip`. To improve fine-grained match quality — since
   every item is the same broad garment type and differences are in fabric,
   weave, print, and border detail — each image is embedded **twice**: once
   as the full photo, once as a center crop (which isolates the fabric
   pattern from background/pose), and the two embeddings are averaged. This
   materially improves discrimination over plain full-image CLIP embeddings.
2. **Indexing**: All embeddings are stored in **ChromaDB**, a local vector
   database, with cosine similarity as the distance metric.
3. **Agent**: A single tool, `search_similar_sarees`, is exposed to Gemini
   using its native function-calling schema (a JSON `FunctionDeclaration`
   with a typed input/output contract). The model decides when to call it
   based on the conversation — e.g. it won't search if no image has been
   uploaded yet, and will ask for one instead. The call itself goes straight
   to Google's REST API (`generateContent`) rather than through a wrapper
   library — see *Design decisions* below for why.
4. **Frontend**: Streamlit handles image upload, chat interface, and renders
   results as an image grid with similarity scores and product links.

## Tech stack

- **Vector DB**: ChromaDB
- **Embeddings**: CLIP (ViT-B/32, via `open_clip`)
- **Agent / function-calling**: Gemini's native tool-calling, called directly
  via REST (`requests`) against `generativelanguage.googleapis.com`
- **LLM**: `gemini-flash-latest` (Google AI Studio — free tier). This is an
  alias Google maintains to always point at their current recommended Flash
  model, chosen deliberately over pinning a dated version — see below.
- **Frontend**: Streamlit

## Setup

```bash
pip install -r requirements.txt
```

You also need:
1. A pre-built `chroma_db/` folder (see Data Pipeline below) placed in the
   project root.
2. A free Google AI Studio API key set as an environment variable:
   ```bash
   export GOOGLE_API_KEY=your-key-here
   ```
   Get one at [aistudio.google.com](https://aistudio.google.com) — no
   payment method required for the free tier.

   If deploying on Streamlit Community Cloud, add it under
   **Settings → Secrets** as:
   ```toml
   GOOGLE_API_KEY = "your-key-here"
   ```

## Running locally

```bash
streamlit run app.py
```

## Data pipeline (build the index yourself)

The catalogue images and embeddings are not regenerated at app startup —
they're built once, offline, and shipped as a `chroma_db/` folder:

1. `01_download_dataset.py` — downloads all catalogue images from the
   source CSV (`sarees.csv`)
2. `02_embed_and_index.py` — generates CLIP embeddings (full image +
   center crop, averaged) and indexes them into ChromaDB

Both scripts are designed to run in Google Colab (for free GPU access).
After running them, download the resulting `chroma_db/` folder and place
it alongside `app.py` before deploying.

## Deployment

Deployed on **Streamlit Community Cloud**:
1. Push this repo (including the `chroma_db/` folder) to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo
3. Set `GOOGLE_API_KEY` under app secrets
4. Deploy

## Design decisions & trade-offs

- **Why CLIP over a fine-tuned model**: no labeled similarity pairs exist
  for this catalogue, so supervised fine-tuning wasn't feasible in the
  timeframe. CLIP's pretrained visual embeddings already capture texture,
  colour, and pattern well; the full+crop fusion recovers most of the
  fine-grained signal a fine-tuned model would add.
- **Why ChromaDB over Pinecone/Qdrant**: catalogue size (~1,000 items) is
  small enough that a local, file-based vector store is simpler to deploy
  and version — no external service or API key dependency for the vector
  store itself.
- **Why a direct REST call instead of the LangChain/SDK wrapper**: initial
  development used `langchain_google_genai`. Google recently rolled out a
  new API key format (prefixed `AQ.` instead of the older `AIzaSy` format),
  and the installed wrapper/SDK versions mishandled it, surfacing as a
  persistent, misleading `404 NotFound` regardless of which model was
  specified. Calling `generateContent` directly via `requests` — mirroring
  exactly what Google's own REST docs show — sidesteps the wrapper entirely
  and made the failure mode (and eventual fix) immediately legible instead
  of hidden behind several layers of retry/wrapper abstraction.
- **Why `gemini-flash-latest` instead of a pinned version**: Google
  deprecated `gemini-2.5-flash` for new API keys mid-project (returned a
  `404` with an explicit "no longer available to new users" message). Using
  Google's self-updating `-latest` alias avoids pinning to a specific
  snapshot that can be deprecated without notice.
- **Known limitation**: 4 of the 1,074 source images returned 404s from the
  retailer's CDN and were excluded from the index (1,070 indexed).
- **Known limitation**: duplicate SKUs in the source data (158 SKUs appear
  more than once, likely colour variants sharing a SKU) were preserved as
  separate catalogue entries with suffixed IDs, rather than deduplicated,
  since each had a distinct image.
