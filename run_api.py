"""Run the FastAPI backend. Open http://localhost:8000/docs for Swagger UI."""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("UVICORN_RELOAD", "false").lower() == "true"
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=reload_enabled)
