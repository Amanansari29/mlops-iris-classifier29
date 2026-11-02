# Docker Deployment

The Iris classifier is packaged as a small HTTP application. The production
image uses a multi-stage build; the training dataset and training scripts are
used to create the model during the image build and are not copied into the
final runtime stage.

## Files and dependency versions

- `Dockerfile` builds the model and then copies only the installed serving
  dependencies, model artifacts, and prediction server into the runtime image.
- `Dockerfile.single-stage` is a comparison image that retains the training
  data and code in its final filesystem.
- `requirements-serving.txt` pins Python package versions so model
  serialization and inference use the same scikit-learn version.
- `src/train_final_model.py` trains and saves the classifier and label encoder.
- `src/serve.py` serves `/health` and `/predict`.

The Docker builder generates its own Iris CSV through the existing pipeline.
It does not rely on `data/processed/iris_features.csv`, which is ignored by
Git. A clean clone therefore has all necessary build inputs.

## Build and compare images

Run these commands from the repository root with Docker Desktop or Docker
Engine running:

```bash
docker build --tag iris-classifier:v1 .
docker build --file Dockerfile.single-stage --tag iris-classifier:single-stage .
docker image ls iris-classifier
docker image inspect --format='{{.Size}} bytes' iris-classifier:v1
docker image inspect --format='{{.Size}} bytes' iris-classifier:single-stage
```

The GitHub Actions `container-smoke` job builds both tags and prints their
sizes in the Actions log. Record the observed sizes from the run when reporting
results; image sizes depend on the base-image digest and platform, so this
document does not claim an unmeasured size.

Both images contain Python, NumPy, pandas, SciPy, scikit-learn, and joblib, so
the final image should not be expected to be only tens of megabytes. The
multi-stage image's size advantage is the exclusion of raw/processed CSVs and
training code. The comparison build uses the same no-cache dependency install,
so it does not artificially retain a pip download cache or compiler toolchain
that neither build actually uses.

## Run the application

Publish the service only on the local machine by binding the host port to
loopback:

```bash
docker run --detach --rm \
  --name iris-classifier-container \
  --publish 127.0.0.1:8000:8000 \
  iris-classifier:v1
```

The server listens on port `8000` inside the container. `EXPOSE 8000` records
that port in image metadata; `--publish` makes it accessible from the host.
Binding the host side to `127.0.0.1` avoids exposing the sample service to
other machines on the network.

## API checks

Check the health endpoint:

```bash
curl --include http://localhost:8000/health
```

Expected JSON body:

```json
{"status": "healthy"}
```

Send a prediction request with the four measurements and three engineered
features:

```bash
curl --include --request POST http://localhost:8000/predict \
  --header "Content-Type: application/json" \
  --data '{
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2,
    "sepal_area": 17.85,
    "petal_area": 0.28,
    "sepal_to_petal_length_ratio": 3.642857
  }'
```

The response contains a `prediction` of `setosa`, `versicolor`, or `virginica`
and three class probabilities that sum to approximately one.

Requests with missing, non-numeric, non-finite, or oversized feature values are
rejected with a JSON error and an appropriate 4xx status. For example, a body
containing only one feature returns HTTP 400 and lists the missing feature
names.

The Docker health check calls `/health` every 30 seconds, with a five-second
timeout, a ten-second start period, and three retries. It reports whether the
HTTP server can respond, not whether model accuracy has changed since training.
The image runs the serving process as a non-root user.

## Inspect and stop

```bash
docker ps
docker inspect --format='{{.State.Health.Status}}' iris-classifier-container
docker logs iris-classifier-container
docker stop iris-classifier-container
```

Because the run command uses `--rm`, Docker removes the container after it is
stopped. The image remains available until removed explicitly.

## Validation status

The Python unit tests cover model artifact creation and the HTTP API, including
health, valid predictions, and invalid requests. The CI `container-smoke` job
additionally builds the real image, waits for Docker health to pass, and
exercises the service through its published local port. Run Docker locally to
inspect platform-specific image sizes and Docker Desktop status.
