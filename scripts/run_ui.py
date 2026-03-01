from pathlib import Path
from vestibular.ui.app_gradio import build_app

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=[
            str(project_root / "data"),
            str(project_root / "dataset"),
        ],
    )
