"""Chapter 5: launch the minimal HTTP/SSE server with a random tiny model."""

import uvicorn

from examples_05_app import build_app_and_service


app, _service = build_app_and_service()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
