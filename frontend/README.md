# Mortgage RAG Frontend

This Vite/React app calls the existing Gradio `index_pdfs` and `chat_fn` actions through `@gradio/client`. The root `vercel.json` defines the Vercel Services deployment: the frontend is public at `/`, and the Python/Gradio service is public at `/api`.

## Run locally

```sh
cd frontend
npm install
npm run dev
```

For local development with a separately running Gradio backend, open the settings button and enter its public base URL. You can also preconfigure an external backend in `frontend/.env.local`:

```dotenv
VITE_GRADIO_URL=https://your-gradio-backend.example.com
```

## Deploy to Vercel

1. Import this repository into a Vercel project with Services enabled.
2. Keep the project root at the repository root so Vercel reads `vercel.json`.
3. Add `GROQ_API_KEY` to the Vercel project environment for the Python service.
4. Deploy. The React client uses the same deployment's `/api` path by default; set `VITE_GRADIO_URL` only if you intentionally use a different backend URL.

The browser calls `/api` through the public rewrite, so there is no service binding: Vercel bindings are available to server-side functions, while this Vite app makes its Gradio requests in the browser. The backend is public under `/api`; the frontend handles all other paths.

**Runtime limitation:** the Python app stores the indexed documents in process memory, and loads a large embedding model and ML dependencies. Vercel Functions can run multiple ephemeral instances, so an upload and a later question are not guaranteed to reach the same in-memory index; model downloads, package size, and memory limits may also prevent a reliable deployment. The service routing is configured, but this stateful RAG backend may need a persistent Python/container host for dependable use. Hugging Face Spaces is not required.

## Build

```sh
npm run build
```