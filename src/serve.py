"""HTTP server for predictions from the containerized Iris classifier."""

import json
import logging
import math
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "models"))
MODEL_PATH = MODEL_DIR / "final_model.joblib"
ENCODER_PATH = MODEL_DIR / "label_encoder.joblib"
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "8000"))
MAX_REQUEST_BYTES = 16_384
FEATURE_ORDER = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "sepal_area",
    "petal_area",
    "sepal_to_petal_length_ratio",
]
logger = logging.getLogger(__name__)


class PredictionHandler(BaseHTTPRequestHandler):
    """Serve health checks and validated prediction requests."""

    model: ClassVar[RandomForestClassifier | None] = None
    label_encoder: ClassVar[LabelEncoder | None] = None

    def _respond(self, status: int, payload: dict[str, object]) -> None:
        response = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._respond(200, {"status": "healthy"})
            return
        self._respond(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._respond(404, {"error": "not found"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._respond(400, {"error": "Invalid Content-Length"})
            return

        if content_length <= 0:
            self._respond(400, {"error": "Request body is empty"})
            return
        if content_length > MAX_REQUEST_BYTES:
            self._respond(413, {"error": "Request body is too large"})
            return

        try:
            body = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._respond(400, {"error": "Invalid JSON"})
            return

        if not isinstance(body, dict):
            self._respond(400, {"error": "JSON body must be an object"})
            return

        missing = [name for name in FEATURE_ORDER if name not in body]
        if missing:
            self._respond(
                400,
                {"error": "Missing features", "missing": missing},
            )
            return

        raw_values = [body[name] for name in FEATURE_ORDER]
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in raw_values
        ):
            self._respond(
                400,
                {"error": "Feature values must be finite numbers"},
            )
            return
        try:
            values = [float(value) for value in raw_values]
        except OverflowError:
            self._respond(
                400,
                {"error": "Feature values must be finite numbers"},
            )
            return
        if not all(math.isfinite(value) for value in values):
            self._respond(
                400,
                {"error": "Feature values must be finite numbers"},
            )
            return

        if self.model is None or self.label_encoder is None:
            self._respond(503, {"error": "Model is not loaded"})
            return

        features = pd.DataFrame([values], columns=FEATURE_ORDER)
        predicted_index = self.model.predict(features)[0]
        prediction = self.label_encoder.inverse_transform([predicted_index])[0]
        probabilities = self.model.predict_proba(features)[0].tolist()
        self._respond(
            200,
            {
                "prediction": prediction,
                "probabilities": probabilities,
            },
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    PredictionHandler.model = joblib.load(MODEL_PATH)
    PredictionHandler.label_encoder = joblib.load(ENCODER_PATH)

    with ThreadingHTTPServer((HOST, PORT), PredictionHandler) as server:
        logger.info("Serving model on http://%s:%d", HOST, PORT)
        server.serve_forever()


if __name__ == "__main__":
    main()
