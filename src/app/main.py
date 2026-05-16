from fastapi import FastAPI
from fastapi.responses import JSONResponse

from .routes import training

app = FastAPI(
    title="Comment Intent API",
    version="1.0.0",
    description="ML service for comment intent classification"
)

# Include routers
app.include_router(training.router)

@app.get("/")
async def health():
    """
    Health check
    """
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
        }
    )