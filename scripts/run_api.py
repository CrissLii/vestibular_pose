"""Launch the FastAPI backend server."""
import uvicorn
from vestibular.api.server import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "vestibular.api.server:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
