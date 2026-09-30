import React, { useEffect, useRef, useState } from "react";
import { Client, prepare_files } from "@gradio/client";
import {
  ArrowDownToLine,
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  CircleHelp,
  FileText,
  FolderOpen,
  LoaderCircle,
  MessageSquareText,
  Plus,
  RotateCw,
  Settings2,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";

const examples = [
  "What is the interest rate and loan amount?",
  "Which fees are listed on the lender sheet?",
  "What property address is on the deed?",
];

const initialMessages = [
  {
    role: "assistant",
    content: "Your documents are ready when you are. Ask a question and I’ll trace the answer back to its source.",
  },
];

function normalizeUrl(value) {
  return value.trim().replace(/\/+$/, "");
}

function getInitialUrl() {
  return import.meta.env.VITE_GRADIO_URL || "";
}

function App() {
  const [backendUrl, setBackendUrl] = useState(getInitialUrl);
  const [urlDraft, setUrlDraft] = useState(getInitialUrl);
  const [client, setClient] = useState(null);
  const [connection, setConnection] = useState("disconnected");
  const [messages, setMessages] = useState(initialMessages);
  const [question, setQuestion] = useState("");
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState("");
  const [indexedCount, setIndexedCount] = useState(0);
  const [sources, setSources] = useState("");
  const [retrievalInfo, setRetrievalInfo] = useState("");
  const [docTypes, setDocTypes] = useState(["All"]);
  const [docType, setDocType] = useState("All");
  const [topK, setTopK] = useState(4);
  const [autoRoute, setAutoRoute] = useState(true);
  const [busy, setBusy] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [error, setError] = useState("");
  const conversationRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!backendUrl) {
      setConnection("disconnected");
      setClient(null);
      return undefined;
    }

    let active = true;
    setConnection("connecting");
    setClient(null);
    Client.connect(normalizeUrl(backendUrl))
      .then((connectedClient) => {
        if (!active) return;
        setClient(connectedClient);
        setConnection("connected");
        setError("");
      })
      .catch((connectError) => {
        if (!active) return;
        setConnection("error");
        setError(connectError.message || "Could not connect to the Gradio backend.");
      });

    return () => {
      active = false;
    };
  }, [backendUrl]);

  useEffect(() => {
    conversationRef.current?.scrollTo({
      top: conversationRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages, busy]);

  function connectBackend(event) {
    event.preventDefault();
    const nextUrl = normalizeUrl(urlDraft);
    if (!nextUrl) {
      setError("Enter the public URL of your running Gradio backend.");
      return;
    }
    try {
      const parsedUrl = new URL(nextUrl);
      if (!['http:', 'https:'].includes(parsedUrl.protocol)) throw new Error();
    } catch {
      setError("Use a valid http or https backend URL.");
      return;
    }
    setError("");
    setBackendUrl(nextUrl);
    setSettingsOpen(false);
  }

  async function indexDocuments() {
    if (!client || files.length === 0 || busy) return;
    setBusy(true);
    setError("");
    setStatus("Uploading PDFs and building the index...");
    try {
      const fileData = await prepare_files(files);
      const uploadedFiles = await client.upload(fileData, client.config.root);
      const result = await client.predict("/index_pdfs", [uploadedFiles]);
      const [nextStatus, chunkCount, dropdownUpdate] = result.data;
      setStatus(String(nextStatus || "Index built."));
      setIndexedCount(Number(chunkCount) || 0);
      const nextTypes = dropdownUpdate?.choices || ["All"];
      setDocTypes(nextTypes);
      setDocType(dropdownUpdate?.value || "All");
      setFiles([]);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (indexError) {
      setStatus("");
      setError(indexError.message || "PDF indexing failed.");
    } finally {
      setBusy(false);
    }
  }

  async function askQuestion(value = question) {
    const cleanQuestion = value.trim();
    if (!cleanQuestion || !client || busy) return;
    const nextMessages = [...messages, { role: "user", content: cleanQuestion }];
    setMessages(nextMessages);
    setQuestion("");
    setBusy(true);
    setError("");
    try {
      const history = messages.filter((message) => message.role !== "system");
      const result = await client.predict("/chat_fn", [cleanQuestion, history, topK, docType, autoRoute]);
      const [updatedMessages, nextSources, nextInfo] = result.data;
      setMessages(updatedMessages || nextMessages);
      setSources(String(nextSources || ""));
      setRetrievalInfo(String(nextInfo || ""));
    } catch (chatError) {
      setMessages([...nextMessages, {
        role: "assistant",
        content: "I couldn’t reach the retrieval service. Check the backend connection and try again.",
      }]);
      setError(chatError.message || "The question could not be sent.");
    } finally {
      setBusy(false);
    }
  }

  function clearConversation() {
    setMessages(initialMessages);
    setSources("");
    setRetrievalInfo("");
    setError("");
  }

  const sourceLines = sources.split("\n").filter((line) => line && line !== "No sources.");
  const isConnected = connection === "connected";

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="Mortgage RAG home">
          <span className="brand-mark"><BookOpen size={19} strokeWidth={2.2} /></span>
          <span className="brand-name">MORTGAGE<span> / </span>DESK</span>
        </a>
        <div className="topbar-right">
          <div className={`connection-pill ${connection}`}>
            {isConnected ? <Wifi size={14} /> : connection === "connecting" ? <LoaderCircle size={14} className="spin" /> : <WifiOff size={14} />}
            <span>{connection === "connected" ? "Backend connected" : connection === "connecting" ? "Connecting" : connection === "error" ? "Connection issue" : "Backend not set"}</span>
          </div>
          <button className="icon-button settings-trigger" onClick={() => setSettingsOpen((open) => !open)} aria-label="Backend settings" title="Backend settings">
            <Settings2 size={18} />
          </button>
        </div>
      </header>

      <main className="workspace">
        <aside className="library-panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">YOUR WORKSPACE</span>
              <h1>Document library</h1>
            </div>
            <button className="icon-button small" onClick={() => fileInputRef.current?.click()} aria-label="Add PDF documents" title="Add PDF documents"><Plus size={17} /></button>
          </div>

          <input ref={fileInputRef} className="visually-hidden" type="file" accept="application/pdf,.pdf" multiple onChange={(event) => setFiles(Array.from(event.target.files || []))} />
          <button className={`upload-zone ${files.length ? "has-files" : ""}`} onClick={() => fileInputRef.current?.click()}>
            {files.length ? <FileText size={21} /> : <ArrowDownToLine size={21} />}
            <span className="upload-title">{files.length ? `${files.length} PDF${files.length === 1 ? "" : "s"} selected` : "Add PDF documents"}</span>
            <span className="upload-note">{files.length ? files.map((file) => file.name).join(", ") : "Select one or more files to index"}</span>
          </button>
          <button className="index-button" onClick={indexDocuments} disabled={!isConnected || !files.length || busy}>
            {busy && status ? <LoaderCircle size={16} className="spin" /> : <Sparkles size={16} />}
            Build document index
          </button>
          {status && <div className="index-status"><Check size={14} />{status}</div>}

          <div className="library-divider" />
          <div className="section-label"><span>INDEX STATUS</span><span className="status-dot" /></div>
          <div className="index-summary">
            <FolderOpen size={17} />
            <div><strong>{indexedCount ? `${indexedCount} chunks indexed` : "No index built"}</strong><span>{indexedCount ? "Available for this session" : "Upload PDFs to get started"}</span></div>
          </div>

          <div className="library-footnote"><ShieldCheck size={14} /><span>Documents are processed by your connected backend.</span></div>
          <div className="library-bottom"><span>RETRIEVAL</span><span>RAG · GROQ</span></div>
        </aside>

        <section className="chat-panel" aria-label="Document assistant chat">
          <div className="chat-heading">
            <div className="assistant-avatar"><MessageSquareText size={19} /></div>
            <div className="chat-title"><h2>Document assistant</h2><span>Answers grounded in your indexed files</span></div>
            <button className="text-icon-button clear-button" onClick={clearConversation} title="Clear conversation"><Trash2 size={15} /><span>Clear</span></button>
          </div>

          <div className="conversation" ref={conversationRef}>
            <div className="date-divider"><span>SESSION</span></div>
            {messages.map((message, index) => (
              <article className={`message ${message.role === "user" ? "user-message" : "assistant-message"}`} key={`${index}-${message.role}`}>
                {message.role !== "user" && <div className="message-avatar"><Sparkles size={14} /></div>}
                <div className="message-body">
                  <div className="message-meta">{message.role === "user" ? "YOU" : "ASSISTANT"}</div>
                  <div className="message-content">{message.content}</div>
                </div>
              </article>
            ))}
            {busy && !status && <div className="thinking"><span /><span /><span />Searching your indexed documents</div>}
            {!isConnected && messages.length === 1 && (
              <div className="connect-hint"><CircleHelp size={17} /><span>Connect a running Gradio backend using the settings button to start asking questions.</span></div>
            )}
          </div>

          <div className="composer-wrap">
            {messages.length === 1 && (
              <div className="example-prompts">
                {examples.map((example) => <button key={example} onClick={() => { setQuestion(example); if (isConnected) askQuestion(example); }} disabled={!isConnected || busy}>{example}</button>)}
              </div>
            )}
            {error && <div className="error-banner"><span>{error}</span><button onClick={() => setError("")} aria-label="Dismiss error"><X size={14} /></button></div>}
            <form className="composer" onSubmit={(event) => { event.preventDefault(); askQuestion(); }}>
              <textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); askQuestion(); } }} placeholder={isConnected ? "Ask about a rate, fee, balance, or property detail..." : "Connect your backend to ask a question"} rows={1} disabled={!isConnected || busy} aria-label="Ask a question" />
              <button type="submit" className="send-button" disabled={!isConnected || busy || !question.trim()} aria-label="Send question" title="Send question">{busy && !status ? <LoaderCircle size={17} className="spin" /> : <ArrowUp size={18} />}</button>
            </form>
            <div className="composer-caption"><span>Answers may be incomplete. Verify details against source documents.</span><span>ENTER TO SEND · SHIFT + ENTER FOR NEW LINE</span></div>
          </div>
        </section>

        <aside className="evidence-panel">
          <div className="panel-heading evidence-heading">
            <div><span className="eyebrow">ANSWER CONTEXT</span><h2>Evidence & controls</h2></div>
            <SlidersHorizontal className="heading-icon" size={17} />
          </div>
          <div className="control-group">
            <label htmlFor="doc-filter">DOCUMENT FILTER</label>
            <div className="select-wrap"><select id="doc-filter" value={docType} onChange={(event) => setDocType(event.target.value)} disabled={!isConnected}>{docTypes.map((type) => <option key={type} value={type}>{type}</option>)}</select><ChevronDown size={15} /></div>
          </div>
          <div className="control-group">
            <div className="control-label-row"><label htmlFor="top-k">RETRIEVAL DEPTH</label><span className="range-value">{topK}</span></div>
            <input id="top-k" className="range-input" type="range" min="1" max="8" step="1" value={topK} onChange={(event) => setTopK(Number(event.target.value))} />
            <div className="range-labels"><span>Focused</span><span>Broader</span></div>
          </div>
          <label className="toggle-row">
            <span><strong>Auto-route by topic</strong><small>Choose likely document type</small></span>
            <input type="checkbox" checked={autoRoute} onChange={(event) => setAutoRoute(event.target.checked)} />
            <span className="toggle-track" />
          </label>

          <div className="evidence-divider" />
          <div className="section-label"><span>RETRIEVED SOURCES</span><span className="source-count">{sourceLines.length || "—"}</span></div>
          <div className="sources-list">
            {sourceLines.length ? sourceLines.map((source, index) => <div className="source-item" key={`${source}-${index}`}><span className="source-index">{String(index + 1).padStart(2, "0")}</span><span>{source.replace(/^\[\d+\]\s*/, "")}</span></div>) : <div className="empty-evidence"><FileText size={20} /><span>Sources used for your next answer will appear here.</span></div>}
          </div>

          <div className="retrieval-meta">
            <div className="meta-heading"><span>RETRIEVAL INFO</span><button onClick={() => { setSources(""); setRetrievalInfo(""); }} aria-label="Clear retrieval info" title="Clear evidence"><RotateCw size={14} /></button></div>
            <p>{retrievalInfo || "Waiting for a question"}</p>
          </div>
          <div className="evidence-note"><ShieldCheck size={14} /><span>Responses are generated from retrieved document passages.</span></div>
        </aside>
      </main>

      {settingsOpen && <div className="settings-scrim" onMouseDown={(event) => { if (event.target === event.currentTarget) setSettingsOpen(false); }}>
        <section className="settings-dialog" aria-labelledby="settings-title">
          <div className="settings-dialog-head"><div><span className="eyebrow">CONNECTION</span><h2 id="settings-title">Backend settings</h2></div><button className="icon-button" onClick={() => setSettingsOpen(false)} aria-label="Close settings"><X size={18} /></button></div>
          <p className="settings-copy">Vercel serves this interface. A separate, publicly reachable Gradio backend must run your retrieval pipeline and model.</p>
          <form onSubmit={connectBackend}>
            <label className="settings-label" htmlFor="backend-url">GRADIO BACKEND URL</label>
            <input id="backend-url" className="url-input" type="url" value={urlDraft} onChange={(event) => setUrlDraft(event.target.value)} placeholder="https://your-rag-backend.example.com" />
            <div className="settings-help">Use the base URL of your running Gradio app. For Vercel, set <code>VITE_GRADIO_URL</code> in Project Settings to preconfigure this value.</div>
            {error && <p className="settings-error">{error}</p>}
            <button className="index-button settings-submit" type="submit"><Wifi size={16} />Connect backend</button>
          </form>
        </section>
      </div>}
      <footer className="mobile-footer"><button onClick={() => setSettingsOpen(true)}><Settings2 size={15} />Backend</button><button onClick={() => document.querySelector(".evidence-panel")?.scrollIntoView({ behavior: "smooth" })}><SlidersHorizontal size={15} />Evidence</button></footer>
    </div>
  );
}

export default App;