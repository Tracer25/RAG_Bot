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

1. Create a **public** Space and select **Gradio** as the SDK.
2. Sync this repository's files to the Space repository. To deploy automatically on GitHub pushes, configure Hugging Face's [GitHub Actions sync](https://huggingface.co/docs/hub/spaces-github-actions); otherwise, push the files to the Space's Git repository.
3. Keep `app.py` as the app file. The Space reads Python dependencies from `requirements.txt` and system packages from `packages.txt`.
4. In the Space settings, add `GROQ_API_KEY` as a secret. For GitHub Actions sync, also add an `HF_TOKEN` GitHub Actions secret with write access to the Space.

The app binds to `0.0.0.0` and uses the platform-provided `PORT` value, defaulting to Gradio's port `7860`.

The embedding model is downloaded when first needed. Allow time for the initial startup and model download; use a Space with enough memory if model loading fails.
