---
title: Mortgage RAG Assistant
sdk: gradio
app_file: app.py
python_version: "3.11"
---

# Mortgage RAG Assistant

## Deploy

This is a Gradio server, not a Vercel serverless app. Vercel does not run the long-lived Gradio server this application needs, and the PyTorch and sentence-transformers dependencies exceed the practical limits of a typical Vercel Function.

Deploy it on Hugging Face Spaces:

1. Create a new Space and select **Gradio** as the SDK.
2. Connect the Space to this GitHub repository and select the branch you want to deploy.
3. Keep `app.py` as the app file. The Space reads Python dependencies from `requirements.txt` and system packages from `packages.txt`.
4. In the Space settings, add `GROQ_API_KEY` as a secret.

The app binds to `0.0.0.0` and uses the platform-provided `PORT` value, defaulting to Gradio's port `7860`.

The embedding model is downloaded when first needed. Allow time for the initial startup and model download; use a Space with enough memory if model loading fails.
