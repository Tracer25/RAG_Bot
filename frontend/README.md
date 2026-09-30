# Mortgage RAG Frontend

This Vite/React app is a static frontend for the repository's existing Gradio application. It calls the `index_pdfs` and `chat_fn` Gradio API actions through `@gradio/client`. It does not include or replace the Python backend.

## Run locally

```sh
cd frontend
npm install
npm run dev
```

Open the settings button in the app and enter the base URL of a running, publicly reachable Gradio backend. To preconfigure the URL, create `frontend/.env.local`:

```dotenv
VITE_GRADIO_URL=https://your-gradio-backend.example.com
```

## Deploy the frontend to Vercel

1. Import this repository in Vercel.
2. Set the project **Root Directory** to `frontend`.
3. Add `VITE_GRADIO_URL` with the public base URL of your Gradio backend, then deploy.

Vercel serves only the built frontend. The Gradio backend must run separately on a host that supports the app's Python, model, and system dependencies. Hugging Face Spaces is not required; use a suitable Python/container host such as a VM or a container service. Keep `GROQ_API_KEY` on the backend and never add it as a `VITE_` variable.

The browser connects directly to Gradio. The backend must be reachable over HTTPS from the deployed Vercel site, and its host must allow browser requests from that site. Uploaded PDFs and the in-memory index remain on the backend; the current Python app keeps its index in process memory, so a backend restart clears it.

## Build

```sh
npm run build
```