"""Chapter 7: launch HTTP serving with paged KV cache."""

import uvicorn

from examples_07_app import build_app_and_service


app, _service = build_app_and_service()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
