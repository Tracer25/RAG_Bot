import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import re
import fitz
import sys
import numpy as np
import torch
import gradio as gr
import pytesseract
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from groq import Groq

try:
   import faiss
except Exception:
   faiss = None

#Configuration section

EMBED_MODEL_NAME = "intfloat/e5-large-v2"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

device = 'cuda' if torch.cuda.is_available () else 'cpu'
print(f"Using device:{device}")

# Native FAISS wheels can be unstable on newer Python builds; fallback is pure NumPy.
USE_FAISS = (
   faiss is not None
   and os.getenv("RAG_USE_FAISS", "0") == "1"
   and sys.version_info < (3, 13)
)

# Lazily initialize heavy models so the UI can still start even when downloads fail.
embed_model = None
embed_model_error = ""

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


def get_embed_model():
   global embed_model, embed_model_error
   if embed_model is not None:
       return embed_model
   try:
       embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=device)
       embed_model_error = ""
   except Exception as e:
       embed_model = None
       embed_model_error = str(e)
   return embed_model


# Generation layer
def generate(prompt: str) -> str:
    if client is None:
        return "Error generating response: GROQ_API_KEY is not set."
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a mortgage document assistant. Answer questions using only the provided context. If the information is not in the context, say 'I don't have enough information in the provided documents.' Always cite your sources using [1], [2], etc."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating response: {e}"



# storing each chunk as a dataclass to enable filtering, source tracing, and routing
@dataclass
class Chunk:
   text: str
   file_name: str
   page_start: int
   page_end: int
   doc_type: str
   chunk_id: str


   @property
   def source(self) -> str:
       return f"{self.file_name} | Pages {self.page_start}-{self.page_end}"


class Retriever:
   def __init__(self):
       self.index = None
       self.embeddings = None
       self.chunks: List[Chunk] = []
       self.by_doc_type: Dict[str, List[int]] = {}


   # stores chunks to create embeddings, FAISS Index, and metadata table lookup
   def build(self, chunks: List[Chunk]):
       self.chunks = chunks
       self.by_doc_type = {}
       if not chunks:
           self.index = None
           self.embeddings = None
           return

       model = get_embed_model()
       if model is None:
           self.index = None
           raise RuntimeError(f"Embedding model unavailable: {embed_model_error}")

       texts = ["passage: " + c.text for c in chunks]
       # embeds text
       emb = model.encode(
           texts,
           convert_to_numpy=True,
           normalize_embeddings=True,
           batch_size=64,
           show_progress_bar=True,
       ).astype("float32")

       self.embeddings = emb
       if USE_FAISS:
           self.index = faiss.IndexFlatIP(emb.shape[1])
           self.index.add(emb)
       else:
           self.index = True


       for i, c in enumerate(chunks):
           self.by_doc_type.setdefault(c.doc_type, []).append(i)


   # performs semantic similarity search and retrieves top k chunks
   def search(
       self,
       query: str,
       k: int = 4,
       filter_doc_type: Optional[str] = None,
   ) -> List[Tuple[Chunk, float]]:
       # check if index exists to see if build() was called
       if self.index is None:
           return []

       model = get_embed_model()
       if model is None:
           raise RuntimeError(f"Embedding model unavailable: {embed_model_error}")

       query_text = "query: " + query
       # converts the query to an embedding vector
       q = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
       # FAISS search returns indices of most similar chunks and their distances (scores)
       # multiply k by 4 since we might have to filter by document type later (we need extra candidates). Returns a list of lists
       if USE_FAISS:
           D, I = self.index.search(q, min(max(k * 4, 8), len(self.chunks)))
           #returns a list of tuples representing indicies paired with their scores (excluding invalid indicies)
           pairs = [(int(i), float(s)) for i, s in zip(I[0], D[0]) if i != -1]
       else:
           if self.embeddings is None:
               return []
           scores = np.dot(self.embeddings, q[0])
           candidate_count = min(max(k * 4, 8), len(self.chunks))
           top_idx = np.argsort(-scores)[:candidate_count]
           pairs = [(int(i), float(scores[i])) for i in top_idx]


       if filter_doc_type and filter_doc_type != "All":
           allowed = set(self.by_doc_type.get(filter_doc_type, []))
           pairs = [p for p in pairs if p[0] in allowed]


       top = pairs[:k]
       #returns
       return [(self.chunks[i], max(0.0, min(1.0, s))) for i, s in top]


retriever = Retriever()




# document classification (before pre filtering)
def infer_doc_type(filename: str, text_sample: str) -> str:
   name = filename.lower()
   text = text_sample.lower()


   if "mortgage" in name or any(t in text for t in ["interest rate", "escrow", "borrower", "lender"]):
       return "Mortgage Contract"
   if "fee" in name or any(t in text for t in ["origination fee", "closing cost", "underwriting fee"]):
       return "Lender Fee Sheet"
   if "statement" in name or any(t in text for t in ["beginning balance", "ending balance", "transactions"]):
       return "Bank Statement"
   if "invoice" in name or any(t in text for t in ["invoice", "amount due", "bill to"]):
       return "Invoice"
   if "deed" in name or "title" in text:
       return "Land Deed"
   return "Other"




# open and extract text using PyMuPDF, if text cant be retrieved we fallback on OCR
def extract_text(file_obj):
   doc = fitz.open(file_obj.name)
   pages = []
   for i, page in enumerate(doc):
       text = page.get_text("text").strip()
       if not text:
           pix = page.get_pixmap(dpi=150)
           img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
           text = pytesseract.image_to_string(img)
        #filter out whitespace
       text = re.sub(r"\s+", " ", text).strip()
       pages.append((i + 1, text))
   doc.close()
   return pages




# chunks every 250 words to create an embedding
def chunk_pages(file_obj, chunk_words=250, overlap_words=50) -> List[Chunk]:
   pages = extract_text(file_obj)
   chunks: List[Chunk] = []


   # grab file extension before inference
   file_name = os.path.basename(file_obj.name)


   for page_num, text in pages:
       if not text:
           continue
       words = text.split()
       doc_type = infer_doc_type(file_name, text[:1200])
       stride = max(1, chunk_words - overlap_words)


       chunk_idx = 0
       # each new chunk starts after 200 words have been chunked, then new iteration starts 50 words behind. This helps LLM achieve coherence
       for start in range(0, len(words), stride):
           end = min(len(words), start + chunk_words)
           # joins words back into a string
           snippet = " ".join(words[start:end]).strip()
           if snippet:
               chunks.append(
                   Chunk(
                       text=snippet,
                       file_name=file_name,
                       page_start=page_num,
                       page_end=page_num,
                       doc_type=doc_type,
                       chunk_id=f"{file_name}_p{page_num}_c{chunk_idx}",
                   )
               )
               chunk_idx += 1
           if end == len(words):
               break
   return chunks




# auto routing
def predict_query_doc_type(query: str) -> Tuple[str, float]:
   q = query.lower()
   rules = {
       "Mortgage Contract": ["mortgage", "interest", "apr", "loan", "escrow", "principal"],
       "Lender Fee Sheet": ["fee", "closing cost", "origination", "points", "charges"],
       "Bank Statement": ["balance", "deposit", "withdrawal", "transaction", "account"],
       "Invoice": ["invoice", "amount due", "bill", "subtotal", "total"],
       "Land Deed": ["deed", "title", "ownership", "property"],
   }

  #grabs the best type and score for each
   best_type, best_score = "Other", 0
   for t, kws in rules.items():
       score = sum(1 for kw in kws if kw in q)
       if score > best_score:
           best_type, best_score = t, score


   if best_score == 0:
       return "Other", 0.35
   return best_type, min(0.95, 0.5 + 0.15 * best_score)



#build prompt by attaching context

def build_prompt(query: str, retrieved: List[Tuple[Chunk, float]]) -> str:
   context = "\n\n".join(
       f"[{i+1}] {c.source}\n{c.text}"
       for i, (c, s) in enumerate(retrieved)
   ) or "No relevant content."

   #simple prompt requires less tokens, LLM overthinks and hallucinates less
   return f"""Answer the question using only the context provided. If the answer is not in the context, say "I don't have enough information."

   Context:
   {context}

   Question: {query}

   Answer:"""



#check if history list is a dictionary to pass into the chatbot

def _normalize_history(history):
   if not history:
       return []
   if isinstance(history[0], dict):
       return history


   normalized = []
   for turn in history:
       if isinstance(turn, (list, tuple)) and len(turn) == 2:
           normalized.append({"role": "user", "content": str(turn[0])})
           normalized.append({"role": "assistant", "content": str(turn[1])})
   return normalized



#main entry point for every user query
def chat_fn(message, history, k, doc_filter, auto_route):
   history = _normalize_history(history)

  #this checks if we have built the vector index
  #if we have not self.index = None
   if retriever.index is None:
       history += [
           {"role": "user", "content": message},
           {"role": "assistant", "content": "Upload and index PDFs first."},
       ]
       return history, "No sources.", "Chunks used: 0 | Confidence: 0.00 | Route: none"


   active_filter = doc_filter
   route_note = "manual"

   #predicts document type using confidence levels
   #auto route is used when user does not specify a document type
   if auto_route and doc_filter == "All":
       pred_type, conf = predict_query_doc_type(message)
       if pred_type != "Other" and conf >= 0.70 and pred_type in retriever.by_doc_type:
           active_filter = pred_type
           route_note = f"auto->{pred_type} ({conf:.2f})"
       else:
           route_note = f"auto->all ({conf:.2f})"
   #performs semantic retrieval over chunks
   try:
       results = retriever.search(message, int(k), filter_doc_type=active_filter)
   except Exception as e:
       history += [
           {"role": "user", "content": message},
           {"role": "assistant", "content": f"Retrieval error: {e}"},
       ]
       return history, "No sources.", "Chunks used: 0 | Confidence: 0.00 | Route: error"
   #builds the prompt
   prompt = build_prompt(message, results)


   if not results:
       answer = "I don't have enough information in the provided documents."
   else:
       try:
           #generate an answer
           answer = generate(prompt)
       except Exception as e:
           answer = f"Model request failed: {e}"


   sources = "\n".join(
       f"[{i+1}] {c.source} | {c.doc_type} | chunk_id={c.chunk_id} | score={s:.2f}"
       for i, (c, s) in enumerate(results)
   ) or "No sources."

  #calculate average confidence level for each chunk
   avg_conf = (sum(s for _, s in results) / len(results)) if results else 0.0
   meta = f"Chunks used: {len(results)} | Confidence: {avg_conf:.2f} | Route: {route_note}"

  #takes original message and result, then adds it to chat history
   history += [
       {"role": "user", "content": message},
       {"role": "assistant", "content": answer.strip()},
   ]
   return history, sources, meta



#takes uploaded PDF's
def index_pdfs(files):
  #if no files exist, set dropdown (document filter) to "All" and return no files uplaoded
   if not files:
       return "No files uploaded.", 0, gr.update(choices=["All"], value="All")


   all_chunks: List[Chunk] = []
   try:
       for f in files:
           all_chunks.extend(chunk_pages(f))
       #embeds each chunk and adds it to FAISS (after calling chunk_pages())
       retriever.build(all_chunks)
   except Exception as e:
       return (
           f"Indexing failed: {e}",
           0,
           gr.update(choices=["All"], value="All"),
       )
   #grabs the doc types, adds it to list of choices, then updates dropdown
   doc_types = sorted({c.doc_type for c in all_chunks})
   choices = ["All"] + doc_types
   return (
       f"Indexed {len(all_chunks)} chunks from {len(files)} file(s).",
       len(all_chunks),
       gr.update(choices=choices, value="All"),
   )


UI_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
  --bg-main: #f4f7f3;
  --bg-panel: #ffffff;
  --ink-main: #0d1b1e;
  --ink-subtle: #506066;
  --accent: #0f766e;
  --accent-soft: #d9f6ee;
  --outline: #d5ddd8;
}

.gradio-container {
  background:
    radial-gradient(circle at 5% -5%, #d8f5eb 0%, transparent 34%),
    radial-gradient(circle at 100% 0%, #d6ecff 0%, transparent 30%),
    var(--bg-main);
  font-family: "Space Grotesk", sans-serif;
}

#title-wrap {
  background: linear-gradient(135deg, #0f766e, #1d4ed8);
  color: #f8fffd;
  border-radius: 18px;
  padding: 1rem 1.2rem;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 14px 34px rgba(15, 118, 110, 0.26);
  margin-bottom: 8px;
}

.panel {
  background: var(--bg-panel);
  border: 1px solid var(--outline);
  border-radius: 16px;
}

#chat-box {
  border: 1px solid var(--outline);
  border-radius: 14px;
}

#source-box textarea,
#meta-box textarea {
  font-family: "IBM Plex Mono", monospace;
}
"""


# UI layer (layout containers)
with gr.Blocks(
   title="Mortgage RAG Chatbot",
) as demo:
   with gr.Column(elem_id="title-wrap"):
       gr.Markdown("## Mortgage RAG Assistant")
       gr.Markdown("Upload PDF files, build the index, then ask questions with citations.")

   with gr.Row():
       with gr.Column(scale=2, elem_classes=["panel"]):
           chatbot = gr.Chatbot(height=520, elem_id="chat-box", buttons=["copy"])
           user_input = gr.Textbox(
               placeholder="Ask about rates, fees, purchase price, balances, or deed details...",
               label="Question",
           )
           with gr.Row():
               ask = gr.Button("Ask", variant="primary")
               clear = gr.Button("Clear")

           with gr.Accordion("Retrieval Controls", open=False):
               k_slider = gr.Slider(1, 8, value=4, step=1, label="Top K Chunks")
               auto_route = gr.Checkbox(value=True, label="Auto-route by query intent")
               doc_filter = gr.Dropdown(choices=["All"], value="All", label="Document Filter")

           with gr.Accordion("Evidence", open=True):
               sources_box = gr.Textbox(label="Sources (with confidence)", lines=8, elem_id="source-box")
               meta_box = gr.Textbox(label="Retrieval Info", lines=1, elem_id="meta-box")

       with gr.Column(scale=1, elem_classes=["panel"]):
           gr.Markdown("### Index Documents")
           pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDF Files")
           index_btn = gr.Button("Build Index")
           status = gr.Textbox(label="Status")
           chunk_count = gr.Number(label="Chunks Indexed", precision=0)
           gr.Examples(
               examples=[
                   "What is the purchase price and loan amount for John Q. Smith?",
                   "What is the appraisal fee on the lender fee worksheet?",
                   "What is the property location listed in the warranty deed?",
               ],
               inputs=user_input,
               label="Try an example",
           )

   index_btn.click(index_pdfs, pdf_input, [status, chunk_count, doc_filter])

   ask_event = ask.click(
       chat_fn,
       [user_input, chatbot, k_slider, doc_filter, auto_route],
       [chatbot, sources_box, meta_box],
   )
   user_input.submit(
       chat_fn,
       [user_input, chatbot, k_slider, doc_filter, auto_route],
       [chatbot, sources_box, meta_box],
   ).then(lambda: "", None, user_input)
   ask_event.then(lambda: "", None, user_input)

   clear.click(lambda: ([], "", "", ""), None, [chatbot, sources_box, meta_box, user_input])


LAUNCH_KWARGS = dict(
   debug=True,
   server_name="127.0.0.1",
   theme=gr.themes.Soft(
       primary_hue="teal",
       secondary_hue="blue",
       neutral_hue="stone",
   ),
   css=UI_CSS,
)


def launch_app():
   for port in [7860, 7861, 7862, 8000, 8080]:
       try:
           demo.launch(server_port=port, **LAUNCH_KWARGS)
           return
       except OSError:
           continue
   # Final attempt lets Gradio surface the full error details.
   demo.launch(**LAUNCH_KWARGS)


if __name__ == "__main__":
   launch_app()


#------QUESTIONS---------

#What is the purchase price and loan amount for John Q. Smith? (Blob Sample)
#What is the appraisal fee on the XYZ Lender fee worksheet? (Blob Sample)
#What is the sale price and property location in the warranty deed for Michael T. Harrison? (Deed)
#What is Joe Boe's basic pay and net pay from Unknown and Co.? (Test Blob)
