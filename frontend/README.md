# Mortgage RAG Frontend

This Vite/React app calls the existing Gradio `index_pdfs` and `chat_fn` actions through `@gradio/client`. Vercel builds and serves only this static frontend. Run the Python/Gradio backend separately on a host that supports its ML and system dependencies.

## Run locally

```sh
cd frontend
npm install
npm run dev
```

Open the settings button and enter the base URL of a running, publicly reachable Gradio backend. To preconfigure it locally, create `frontend/.env.local`:

```dotenv
VITE_GRADIO_URL=https://your-gradio-backend.example.com
```

## Deploy the frontend to Vercel

1. Import this repository into Vercel and leave the project root at the repository root.
2. Add `VITE_GRADIO_URL` with the public HTTPS base URL of your Gradio backend.
3. Deploy. The root `vercel.json` installs and builds only `frontend`, then serves `frontend/dist`.

Keep `GROQ_API_KEY` on the backend host; do not put it in a `VITE_` variable. If `VITE_GRADIO_URL` is not configured, enter the backend URL in the app's settings after it loads.

The RAG backend requires PyTorch, Transformers, Sentence Transformers, and Tesseract OCR. These exceed Vercel's 500 MB Function bundle limit, so this repository deliberately does not deploy the Python app as a Vercel Function. Use a persistent Python/container host for it; its in-memory document index is cleared when the backend restarts. Hugging Face Spaces is not required.

## Build

```sh
npm run build
```