# LipRead — REST API Documentation

LipRead provides an asynchronous, production-grade REST API powered by **FastAPI** for visual speech recognition inference.

---

## 1. Service Endpoints Overview

| Method | Endpoint | Description | Auth |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root service index and documentation links | None |
| `GET` | `/health` | Health status and model loaded state | None |
| `POST` | `/predict` | Upload video file and obtain lip-reading transcription | None |
| `GET` | `/docs` | Interactive Swagger UI documentation | None |
| `GET` | `/openapi.json` | OpenAPI 3.1 specification schema | None |

---

## 2. Interactive Documentation

When the FastAPI server is running (default port 8000), interactive documentation is available at:
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)
- **OpenAPI Schema:** [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## 3. Detailed Endpoint Specifications

### `GET /` — Service Index

Returns service metadata, version information, and navigation links.

#### Response (200 OK)
```json
{
  "service": "LipRead Inference API",
  "version": "0.1.0",
  "status": "online",
  "docs_url": "/docs",
  "openapi_url": "/openapi.json",
  "health_url": "/health"
}
```

---

### `GET /health` — Service Health Probe

Indicates whether the API server is healthy and whether the PyTorch model checkpoint is currently loaded in memory.

#### Response (200 OK) — Model Loaded
```json
{
  "status": "ok",
  "service": "LipRead",
  "model_loaded": true,
  "device": "cuda:0",
  "checkpoint": "best_model.pt",
  "model_version": "0.1.0"
}
```

#### Response (200 OK) — Degraded (No Checkpoint Found)
```json
{
  "status": "degraded",
  "service": "LipRead",
  "model_loaded": false,
  "device": null,
  "checkpoint": null,
  "model_version": "0.1.0"
}
```

---

### `POST /predict` — Video Transcription Inference

Accepts an uploaded video clip, validates file format and size, extracts mouth frames via OpenCV and MediaPipe, processes through the 3D CNN + Bi-LSTM neural model, and returns decoded transcripts.

#### Query Parameters
- `decode_method` *(string, optional, default: `"greedy"`)*: Decoding strategy (`"greedy"` or `"beam"`).
- `beam_width` *(integer, optional, default: `10`)*: Beam search candidate width (only applicable when `decode_method="beam"`). Range: 1 to 50.

#### Request Body
- `file` *(binary upload, multipart/form-data, required)*: Video file.
  - **Supported formats:** `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.mpg`, `.mpeg`
  - **Maximum payload size:** 50 MiB

#### Example Request (`curl`)
```bash
curl -X POST "http://localhost:8000/predict?decode_method=greedy" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_utterance.mpg"
```

#### Response (200 OK) — Standard CTC Output
```json
{
  "transcription": "bin blue at f two now",
  "raw_transcription": "bin blue at f two now",
  "corrected_transcription": null,
  "applied_correction": false,
  "correction_model": null,
  "confidence": null,
  "inference_time_ms": 142.5,
  "frame_count": 75,
  "fps": 25.0
}
```

#### Response (200 OK) — With LLM Contextual Correction Enabled
```json
{
  "transcription": "bin blue at f two now",
  "raw_transcription": "pin blue at f two now",
  "corrected_transcription": "bin blue at f two now",
  "applied_correction": true,
  "correction_model": "gpt-4o-mini",
  "confidence": null,
  "inference_time_ms": 142.5,
  "frame_count": 75,
  "fps": 25.0
}
```

---

## 4. HTTP Error Handling & Status Codes

All errors return structured JSON matching the `ErrorResponse` schema:

```json
{
  "detail": "Descriptive explanation of the error.",
  "error_code": "ERROR_IDENTIFIER"
}
```

| HTTP Status | Error Code | Cause / Resolution |
| :--- | :--- | :--- |
| **400 Bad Request** | `EMPTY_FILE` | Uploaded video file is 0 bytes. |
| **400 Bad Request** | `INVALID_PARAMETER` | Unsupported decoding method or invalid beam width. |
| **413 Payload Too Large** | `FILE_TOO_LARGE` | Video file exceeds the 50 MiB payload limit. |
| **415 Unsupported Media Type** | `UNSUPPORTED_MEDIA_TYPE` | Uploaded file extension is not in the allowed video list. |
| **500 Internal Server Error** | `PROCESSING_ERROR` | Video frames could not be decoded or OpenCV/MediaPipe pipeline failed. |
| **503 Service Unavailable** | `MODEL_NOT_LOADED` | No model checkpoint was found at startup. Provide a valid checkpoint. |
