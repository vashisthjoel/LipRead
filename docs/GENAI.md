# LipRead — GenAI Contextual Post-Processing Layer

## 1. Executive Summary & Purpose

Lip reading is inherently affected by **viseme ambiguity**: distinct phonemes share identical visible lip geometries. For example, bilabial plosives (/p/, /b/, /m/) and labiodental fricatives (/f/, /v/) produce visually indistinguishable mouth configurations. Even a converged neural visual speech recognition model can output phonetically transposed transcriptions (e.g. producing `"lay freen in zero soon"` instead of `"lay green in zero soon"`).

The GenAI contextual post-processing layer introduces an LLM-assisted correction stage that consumes raw Connectionist Temporal Classification (CTC) transcriptions, leverages natural language linguistic context to disambiguate visemes, and produces a linguistically coherent corrected transcription.

> **CRITICAL ARCHITECTURAL BOUNDARY:**
>
> **The LLM is a post-processing layer. It does not constitute additional training of the lip-reading model.**
>
> The LLM operates strictly downstream of the neural pipeline. It does **not** replace the 3D CNN, Bi-LSTM, CTC loss, CTC decoder, or `InferenceEngine`. Furthermore, we do not claim that contextual correction improves CER/WER unless experimentally measured and benchmarked on ground-truth evaluation datasets.

---

## 2. End-to-End Architecture

```
                       Video File (.mp4, .avi, etc.)
                                    │
                                    ▼
                     LipPreprocessingPipeline (OpenCV)
                                    │
                                    ▼
                       3D CNN Spatiotemporal Front-End
                                    │
                                    ▼
                         Bidirectional LSTM Backbone
                                    │
                                    ▼
                         CTC Linear Projection Layer
                                    │
                                    ▼
                     CTC Decoder (Greedy or Prefix Beam)
                                    │
                                    ▼
                         Raw CTC Transcription
                                    │
                                    ▼
                ┌───────────────────────────────────────┐
                │        ContextualCorrector            │
                │                                       │
                │   • Feature flag check (GENAI_ENABLED)│
                │   • LangChain PromptTemplate          │
                │   • Configured ChatModel Provider     │
                │   • StrOutputParser & Cleaning        │
                │   • Fault-tolerant exception handling │
                └───────────────────┬───────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
          FastAPI REST API                   Streamlit Frontend
     Returns typed response with:      Visualizes:
     • raw_transcription               • RAW MODEL OUTPUT
     • corrected_transcription         • LLM CONTEXTUAL CORRECTION
     • correction_applied              • Distinction notice
     • correction_model                • Inference performance
```

---

## 3. LangChain Orchestration & Provider Abstraction

LipRead employs **LangChain** (`langchain`, `langchain-core`, `langchain-community`) solely as a decoupled orchestration framework. The core system does **not** hardcode any specific LLM provider or proprietary API SDK.

### Core Abstraction

The processing workflow is defined by the following contract:

$$\text{raw\_transcript} \xrightarrow{\quad\text{ContextualCorrector}\quad} \text{CorrectionResult}(\text{raw}, \text{corrected}, \text{applied}, \dots)$$

### Provider Factory (`src/genai/providers.py`)

A factory pattern (`create_chat_model`) instantiates the requested provider based on runtime configuration:

| Provider | Target Class | Use Case / Notes |
|:---|:---|:---|
| `mock` | `MockChatModel` | Default offline mode; rule-based deterministic viseme correction without network or cost. |
| `openai` | `ChatOpenAI` | Commercial cloud inference (e.g. `gpt-4o-mini`, `gpt-4o`). Requires `LIPREAD_LLM_API_KEY`. |
| `anthropic` | `ChatAnthropic` | Commercial cloud inference (e.g. `claude-3-5-sonnet`). Requires `LIPREAD_LLM_API_KEY`. |
| `ollama` | `ChatOllama` | Self-hosted local inference (e.g. `llama3`, `mistral`). Zero API keys, local privacy. |

---

## 4. Prompt Engineering & Behavioral Constraints

The system prompt in `src/genai/prompts.py` is engineered with explicit guardrails:

1. **Viseme Disambiguation:** Instructs the LLM that the text originates from an automated lip-reading model and may contain phonetic/viseme substitution errors.
2. **Preservation of Meaning:** Mandates preserving the user's intended semantics without creative embellishment.
3. **No Information Fabrication:** Explicitly prohibits adding unsupported facts, entities, or extraneous conversational commentary.
4. **Minimal Edit Distance:** Discourages gratuitous rewrites when the raw text is already grammatical and sensible.
5. **Format Restriction:** Enforces returning **ONLY** the corrected transcript string (no preamble, markdown formatting, or enclosing quotes).

---

## 5. Result Schema & Metadata Integrity

Correction outcomes are encapsulated in the typed `CorrectionResult` dataclass:

```python
@dataclass
class CorrectionResult:
    raw_transcription: str
    corrected_transcription: str
    correction_applied: bool
    provider: Optional[str] = None
    model: Optional[str] = None
    latency_seconds: float = 0.0
    error: Optional[str] = None

    @property
    def effective_text(self) -> str:
        """Returns corrected text if correction applied, otherwise raw."""
        return self.corrected_transcription if self.correction_applied else self.raw_transcription
```

- **Confidence Proxy:** In accordance with architectural principles, confidence scores are never synthesized or fabricated for LLM corrections. Confidence proxies reflect only the raw neural frame log-probabilities from the `InferenceEngine`.
- **Backward Compatibility:** The existing `/predict` response `transcription` field contains `effective_text`, ensuring existing API consumers continue functioning without breaking changes.

---

## 6. Disabled Mode & Fallback Resilience

The system guarantees 100% operational availability even when GenAI is completely disabled or unreachable:

### Disabled State (`GENAI_ENABLED=false` or `use_correction=false`)

When disabled:
- `raw_transcription == corrected_transcription`
- `correction_applied == False`
- `provider == None`, `model == None`
- Zero network overhead, zero latency impact.

### Fault Tolerance & Error Handling

If an external LLM call encounters a network timeout, rate limit, quota exhaustion, or malformed response:
- The exception is captured and logged with a warning.
- The request **does not fail**.
- The endpoint returns `raw_transcription`, with `corrected_transcription = raw_transcription` and `correction_applied = False`.
- The neural lip-reading inference remains completely usable regardless of external LLM availability.

---

## 7. Environment Variables & Security Policies

> **SECURITY MANDATE:**
>
> API keys and credentials must **NEVER** be committed to Git, embedded in source files, included in documentation, or passed through Streamlit UI code.

All settings are resolved via environment variables or the `.env` file (which is git-ignored):

| Variable | Default | Purpose |
|:---|:---|:---|
| `GENAI_ENABLED` | `false` | Master toggle for the GenAI post-processing layer. |
| `LIPREAD_LLM_PROVIDER` | `mock` | Selected provider (`mock`, `openai`, `anthropic`, `ollama`). |
| `LIPREAD_LLM_MODEL` | `gpt-4o-mini` | Model name or tag. |
| `LIPREAD_LLM_API_KEY` | *(None)* | Secret API key (resolved securely from environment). |
| `LIPREAD_LLM_API_BASE` | *(None)* | Custom API base URL (e.g. `http://localhost:11434` for Ollama). |
| `LIPREAD_LLM_TEMPERATURE` | `0.0` | Sampling temperature (0.0 enforces deterministic output). |
| `LIPREAD_LLM_MAX_TOKENS` | `128` | Maximum token budget for output transcription. |
| `LIPREAD_LLM_TIMEOUT` | `15.0` | Client timeout in seconds before fallback activates. |

A safe template with empty placeholders is provided at [`.env.example`](file:///D:/Lipreading%20end%20to%20end%20detection/.env.example).

---

## 8. Testing Strategy

All automated unit and integration tests are **hermetic and offline**:
- Zero real API calls are executed during testing.
- No network access, paid subscription, or active API key is required.
- The `mock` provider and `unittest.mock` fixtures supply deterministic responses.
- Test suite verifies disabled mode, successful viseme correction, whitespace/empty inputs, timeout fallbacks, quote stripping, API route parameters, and client deserialization.

---

## 9. Current Limitations

1. **Acoustic/Visual Ground Truth:** Contextual correction operates purely on the lexical output; it has no access to the raw visual features.
2. **Proper Nouns & Neologisms:** LLMs may inadvertently "correct" rare surnames, slang, or domain-specific acronyms unless fine-tuned or instructed with few-shot context.
3. **Inference Latency:** Invoking remote cloud LLMs introduces network round-trip latency (typically 200ms–1500ms) on top of the ~300ms neural inference. For real-time streaming, local quantized models (e.g. Ollama) or heuristic decoders are recommended.
