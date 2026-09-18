"""Chapter 8: launch HTTP serving with RadixAttention prefix reuse."""

import uvicorn
from examples_08_app import build_app_and_service

app, _service = build_app_and_service()


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
