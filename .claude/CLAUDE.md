# Audiblez Web

## Stack
- **Backend**: Python (file processing pipeline)
- **Frontend**: SvelteKit
- **Processing**: Ebook to audiobook conversion

## Quick Start
```bash
# Backend
cd backend && pip install -r requirements.txt && python main.py

# Frontend
cd frontend && npm install && npm run dev
```

## Architecture
- `backend/` — Python file processing server
- `frontend/` — SvelteKit web UI
- `samples/` — Sample audio files
- `Dockerfile` — Production container
- `Dockerfile.base` — Base image with heavy ML dependencies (pre-built to speed up CI)

## Key Patterns
- **Two-stage Docker**: `Dockerfile.base` bakes slow ML deps, `Dockerfile` layers app code on top
- **File processing**: Upload ebook, process async, stream audio output

## CI/CD
- **Build**: GitHub Actions (Docker image push to DockerHub)
- **Deploy**: Woodpecker CI (kubectl set image)
- **Image tags**: 8-char git SHA
