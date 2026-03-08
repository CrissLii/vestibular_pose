"""Launch the FastAPI backend server."""
import uvicorn
from vestibular.api.server import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "vestibular.api.server:app",
        host="0.0.0.0",  # 对外暴露，允许局域网/外网访问
        port=8000,
        reload=False,
    )
