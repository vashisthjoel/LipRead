# Docker Deployment Guide for LipRead

This directory provides containerization configurations for running the **LipRead** FastAPI backend and Streamlit web application.

---

## 1. Quick Start with Docker Compose

To build and run both services:

```bash
cd docker
docker-compose up --build
```

- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Streamlit Application:** [http://localhost:8501](http://localhost:8501)

---

## 2. Mounting Checkpoints

The container expects a trained PyTorch model checkpoint at `/app/checkpoints/best_model.pt`.

To provide a local checkpoint, mount your directory via volume mapping in `docker-compose.yml`:

```yaml
volumes:
  - /path/to/local/checkpoints:/app/checkpoints:ro
```

If no checkpoint is mounted, the FastAPI service will start in degraded mode (`status: "degraded"`), and the Streamlit frontend will display an alert indicating that a checkpoint is required for inference.

---

## 3. Building Individual Containers

### FastAPI Service Only:
```bash
docker build -t lipread-api -f docker/Dockerfile.api .
docker run -p 8000:8000 lipread-api
```

### Streamlit Application Only:
```bash
docker build -t lipread-app -f docker/Dockerfile.app .
docker run -p 8501:8501 -e LIPREAD_API_URL="http://your-api-host:8000" lipread-app
```
