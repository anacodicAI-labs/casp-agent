import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router

_CODE_DIR = Path(__file__).resolve().parent.parent

# Load .env before anything else reads os.environ.
_ENV_FILE = _CODE_DIR / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)

# Ensure we run from project root so relative data/ imports resolve
if os.getcwd() != str(_CODE_DIR):
    os.chdir(_CODE_DIR)

app = FastAPI(
    title="Supply Chain Option 2 API",
    description="Query -> extract + data-derived defaults -> edit -> run optimization",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# Serve frontend assets (e.g. supply_chain.jpg)
_FRONTEND_DIR = _CODE_DIR / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

app.include_router(router)


 
