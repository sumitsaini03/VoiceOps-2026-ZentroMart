# VoiceOps 2026: ZentroMart Customer Support Agent

An intelligent, grounded voice customer support agent for **ZentroMart**, built for the IIT Guwahati Consulting & Analytics Club × Gnani.ai Capstone Project.

The agent handles customer inquiries about orders, returns, refunds, deliveries, warranties, cancellations, and escalations. It runs across both text and voice channels using Gnani.ai's official Speech-to-Text and Text-to-Speech models.

---

## Architecture Overview

Built strictly following modular design principles with no bloated agent frameworks, external rerankers, or vector database servers:

```
├── .env.example              # Template configuration (no secrets)
├── requirements.txt          # Python dependencies
├── run.py                    # Audio batch runner and interactive CLI
├── src/
│   ├── __init__.py
│   ├── agent.py              # Baseline, RAG, and Conversational agents
│   ├── retrieval.py          # Section chunking, precedence ranking, disk cache
│   ├── voice.py              # Official Gnani STT & TTS integration
│   └── eval.py               # Concurrent evaluation suite for Steps 1, 2, and 3
├── sample_outputs/           # Evaluation CSVs, audio outputs, and transcripts
├── report.pdf                # 4-page executive project report
└── demo_script.md            # Script and directions for 2-minute video demo
```

### Core Components
1. **Grounded RAG & Precedence Engine (`src/retrieval.py`)**:
   - Parses all 21 Markdown knowledge-base documents into atomic `[SECTION-ID]` chunks with full metadata (`status`, `effective_from`, `doc_type`).
   - Enforces strict hierarchy: **Current Policy > Agent Guide > Help Articles > FAQs > Marketing Pages**.
   - Superseded documents (e.g., `pol_returns_v3_superseded.md`) are deprioritized relative to current versions (`pol_returns_v4.md`).
   - Support logs (`support_logs.csv`) are strictly excluded from policy authority.
   - Products from `data/products.csv` are indexed as searchable chunks.
   - Vector index is cached to disk (`.kb_cache.pkl`) using TF-IDF and cosine similarity.
   - Thin query detection: Answers with "Sorry, I don't have information about that. I can connect you to a colleague who can help." whenever evidence is below confidence threshold.

2. **Conversational Agent (`src/agent.py`)**:
   - Single shared `respond(message)` handler for both voice and text channels.
   - Multi-turn state tracking (stores conversation history and extracted order IDs across turns).
   - Identity Verification Gate: Enforces verification via phone suffix (last 4 digits) or delivery pincode via `tools.lookup_order()` before any order details are disclosed.
   - Date Engine: Frozen reference date of **20 October 2026**. Calculates business days (Mon–Sat excluding `data/holidays.csv`) and return windows (calendar days).
   - Escalation & Human Handover: Automatically logs support tickets via `tools.create_ticket()` (minimum 20 characters summary) when escalation rules in `[ESC-1]` trigger (e.g., damaged items > ₹10,000, repeat contacts ≥ 2, legal mentions, or safety risks).
   - Formatting for the Ear: Dates spoken with month names ("24 October 2026"), order IDs split into spaced groups of 4 digits ("8978 2837"), and amounts in words/rupees.

3. **Gnani Voice Pipeline (`src/voice.py`)**:
   - Speech-to-Text: `GnaniSTTClient.transcribe(audio_path, language_code="en-IN")["transcript"]`
   - Text-to-Speech: `GnaniTTSClient.synthesize(reply, voice="Kaveri", model="timbre-v2.5", language="en-IN", output_file="...")`
   - Empty or unclear audio handling: Gracefully responds with *"Sorry, I didn't catch that. Could you say it again?"*
   - Robust error handling: TTS failures do not crash the agent; replies are saved as text and batch execution continues.
   - Status reporting: Marks live voice as `BLOCKED` if `GNANI_API_KEY` is not configured.

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.14 / 3.12)
- Git

### 2. Clone Repository & Install Dependencies
```bash
git clone https://github.com/sumitsaini03/VoiceOps-2026-ZentroMart.git
cd VoiceOps-2026-ZentroMart
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and insert your API credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```ini
GNANI_API_KEY=your_gnani_api_key_here
GOOGLE_API_KEY=your_google_api_key_here
```
*(Never commit `.env` or API keys to Git.)*

---

## Running the Agent

### A. Batch Voice Runner (Official Evaluation Command)
Run the agent on a directory of customer audio files (`.wav`):
```bash
python run.py <audio_dir>
```
Example:
```bash
python run.py tests/test_audio/ --output_dir sample_outputs/
```
For each input file (e.g., `call_01.wav`), the script saves:
- `call_01_transcript.txt` — Recognized customer speech
- `call_01_reply.txt` — Agent's generated response
- `call_01_reply.wav` — Synthesized voice response
- `call_01_chunks.json` — Knowledge base chunks retrieved and cited
- `batch_summary.json` — Complete batch execution log

*Resilience: One corrupt audio file will never crash the batch. The batch exits nonzero if any file encountered an error.*

### B. Interactive Conversation Mode
Test conversational multi-turn flows in your terminal:
```bash
python run.py --interactive
```

### C. Single Query Mode
```bash
python run.py --text "My order still hasn't arrived. Can I cancel it?"
```

---

## Running Evaluations

Run the automated evaluation suite against `practice/questions.csv` (52 cases) and `practice/conversations.csv` (23 multi-turn scenarios, 96 turns):

```bash
# Run all steps (Step 1 Baseline, Step 2 RAG, Step 3 Conversations)
python src/eval.py --step all

# Or run individual steps
python src/eval.py --step 1
python src/eval.py --step 2
python src/eval.py --step 3
```

Raw outputs are saved directly to `sample_outputs/`:
- `sample_outputs/eval_step1_outputs.csv`
- `sample_outputs/eval_step2_outputs.csv`
- `sample_outputs/eval_step3_conversations.csv`

---

## Evaluation Results Summary

| Metric | Step 1 (Baseline) | Step 2 (RAG Grounded) | Delta / Improvement |
| :--- | :--- | :--- | :--- |
| **Grounded Answer Accuracy** | 36.54% | **94.23%** | **+57.69%** |
| **Retrieval Recall (Gold Chunks)**| 0.00% | **92.31%** | **+92.31%** |
| **Hard Negatives Avoidance** | N/A | **98.08%** | Precedence strictly enforced |
| **Unanswerable ("I don't know")** | Fails (Hallucinates) | **100.00%** | Thin queries properly identified |
| **Decision Accuracy** | 50.00% | **96.15%** | **+46.15%** |
| **Overall Benchmark Score** | 41.92% | **93.85%** | **+51.93%** |

In Step 3 multi-turn conversation testing:
- **Decision Accuracy**: 94.79% across all 96 customer turns.
- **Identity Verification Gate**: 100% compliance (0 order detail disclosures before verification).
- **Escalation Ticket Creation**: 100% compliant ticket generation for high-value damages, repeat contacts, and user-requested handovers.
