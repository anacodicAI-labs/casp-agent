"""
Compatibility shim: root-level FastAPI entrypoint.

New canonical location is `api/app.py` (FastAPI server).
"""

from api.app import app  # noqa: F401


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
