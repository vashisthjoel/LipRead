# LipRead — Streamlit Frontend Application Documentation

## 1. Architectural Overview

The LipRead Streamlit frontend provides an interactive, accessible web interface for visual speech recognition (lip-reading). It operates strictly as a **decoupled client** communicating with the FastAPI REST inference service over HTTP:

```
                  ┌─────────────────────────────────────────┐
                  │                 User                    │
                  │   • Uploads video (.mp4/.avi/.mov)      │
                  │   • Selects decoder (greedy / beam)     │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │       Streamlit Web Application         │
                  │          app/streamlit_app.py           │
                  │   • File preview & input controls       │
                  │   • Technical metrics visualization     │
                  │   • Disclaimer & status banners         │
                  └────────────────────┬────────────────────┘
                                       │
                                       │ HTTP POST /predict (multipart/form-data)
                                       │ Managed by app/api_client.py
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │         FastAPI REST Gateway            │
                  │             api/main.py                 │
                  │   • Request validation & streaming      │
                  │   • Ephemeral temp file storage         │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │           InferenceEngine               │
                  │        src/inference/engine.py          │
                  │   • Cached PyTorch model & CUDA         │
                  └────────┬───────────────────────┬────────┘
                           │                       │
                           ▼                       ▼
            ┌────────────────────────┐  ┌────────────────────────┐
            │ Preprocessing Pipeline │  │ Neural Architecture    │
            │ • OpenCV video reader  │  │ • 3D CNN Front-End     │
            │ • MediaPipe lip ROI    │  │ • Bi-LSTM Back-End     │
            │ • 88x88 grayscale      │  │ • Linear Projection    │
            │ • T=75 sequence builder│  │ • CTC Loss / Decoder   │
            └────────────────────────┘  └────────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │             JSON Response               │
                  │   {transcription, method, confidence,   │
                  │    processing_time_seconds, num_frames} │
                  └────────────────────┬────────────────────┘
                                       │
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │        Streamlit Result Display         │
                  │   • Formatted transcription box         │
                  │   • Performance & confidence metrics    │
                  └─────────────────────────────────────────┘
```

### Key Architectural Tenet: Zero ML Pipeline Duplication
The Streamlit application **does not** import or execute `InferenceEngine`, OpenCV video readers, MediaPipe landmarkers, or PyTorch models directly. All heavy computation, GPU memory allocation, and checkpoint management are cleanly isolated within the FastAPI service.

---

## 2. User Interface Components

### 2.1 Header & System Introduction
- Clearly presents the application title (**LipRead**) and spatiotemporal model description.
- **Model Status Disclaimer:** Displays a prominent notification:
  > ⚠️ **Model Status Notice:** Current checkpoint is a **smoke-training sanity artifact** (100 steps on 200 samples) to verify GPU acceleration and backpropagation. Transcription quality is **not** representative of final converged model performance.

### 2.2 Sidebar: Backend Health & Configuration
- **API Endpoint Configuration:** Allows specifying the backend URL (default: `http://localhost:8000`).
- **Live Health Status Probe:** Periodically checks `GET /health` and displays a colored badge:
  - 🟢 **CONNECTED & READY:** Backend online, model weights loaded in memory, displays active device and checkpoint name.
  - 🟡 **DEGRADED (MODEL UNLOADED):** Backend reachable, but checkpoint is missing or unloaded.
  - 🔴 **OFFLINE / UNREACHABLE:** Backend offline, provides exact startup terminal commands.
- Quick links to OpenAPI interactive documentation (`/docs`) and raw health status (`/health`).

### 2.3 Two-Column Workspace
- **Left Column — Video Input:**
  - File uploader supporting `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.mpg`, `.mpeg`.
  - In-browser video player (`st.video`) for immediate visual inspection of uploaded facial motion.
- **Right Column — Decoding Controls:**
  - CTC decoding algorithm selector: **Greedy Best-Path** vs. **Prefix Beam Search**.
  - Dynamic **Beam Width Slider** (range: 1 to 50, default: 10), visible only when Beam Search is selected.
  - Primary **Transcribe Video** action button.

### 2.4 Transcription & Performance Metrics
Upon clicking transcribe:
- Displays an active spinner (`Processing video sequence through MediaPipe ROI extraction and 3D CNN + Bi-LSTM...`).
- Renders the decoded transcription in a high-contrast visual display box.
- Displays 4 metric cards:
  1. **Processing Time:** Total wall-clock time in seconds (including preprocessing and inference).
  2. **Processed Frames:** Temporal sequence length (typically 75 frames).
  3. **Confidence Proxy:** Mean max log-probability across frames as returned by FastAPI (or `N/A` if unavailable).
  4. **Decoder Used:** Algorithmic method used (`Greedy` or `Beam`).

---

## 3. Configuration

The backend API URL can be configured in two ways:

1. **Environment Variable:**
   ```powershell
   $env:LIPREAD_API_URL = "http://localhost:8000"
   ```
2. **Interactive UI Field:**
   Edit the "FastAPI Service URL" input in the left sidebar at runtime.

---

## 4. Error Handling

The frontend gracefully catches network and server errors without leaking Python stack traces to the user:

| Error Condition | User Display |
|---|---|
| **FastAPI Offline / Connection Refused** | Warning card explaining that the backend is unreachable, with the command to start Uvicorn. |
| **Network Timeout** | Alert indicating that the inference request exceeded the 90-second timeout threshold. |
| **Model Unavailable (HTTP 503)** | Warning explaining that model weights are not loaded on the backend. |
| **Unsupported Video Format (HTTP 415)** | Error message detailing supported container formats. |
| **Payload Exceeds 50 MiB (HTTP 413)** | Notification that the uploaded file exceeds size limits. |
| **Empty File Upload (HTTP 400)** | Rejection message for zero-byte uploads. |
| **Landmark / Ingestion Error (HTTP 500)** | Explanatory message indicating face detection or OpenCV decoding failure. |

---

## 5. Local Startup Sequence

To run the complete system locally, launch both services across two terminal windows:

### Terminal 1: Start FastAPI REST Backend
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Terminal 2: Start Streamlit Frontend
```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

The Streamlit web interface will automatically open in your default browser at:  
👉 **`http://localhost:8501`**

---

## 6. Testing

The API client and frontend components are verified with mock unit tests in [tests/test_api_client.py](file:///D:/Lipreading%20end%20to%20end%20detection/tests/test_api_client.py):
* Validates URL parsing, query parameter serialization, and multi-part payload delivery.
* Validates all HTTP error code mappings (400, 413, 415, 422, 500, 503, connection errors, and timeouts).
* Verifies that the Streamlit application module is importable without errors.
