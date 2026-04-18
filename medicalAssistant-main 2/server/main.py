from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
try:
    from server.middlewares.exception_handlers import catch_exception_middleware
    from server.routes.upload_pdfs import router as upload_router
    from server.routes.ask_question import router as ask_router
except ModuleNotFoundError:
    from middlewares.exception_handlers import catch_exception_middleware
    from routes.upload_pdfs import router as upload_router
    from routes.ask_question import router as ask_router



app=FastAPI(title="Medical Assistant API",description="API for AI Medical Assistant Chatbot")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)



# middleware exception handlers
app.middleware("http")(catch_exception_middleware)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Medical Assistant backend is running.",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# routers

# 1. upload pdfs documents
app.include_router(upload_router)
# 2. asking query
app.include_router(ask_router)
