# Datagen Orchestrator

Datagen Orchestrator is a collection of utilities and pipelines for generating synthetic datasets using several AI services. The project includes a command line interface (CLI) that guides you through generating images and annotations and uploading the results to a Datagen dataset.

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd datagen_orchestrator
   ```
2. **Install the Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   Python 3.12 or later is recommended.

## Running the CLI

Execute the interactive CLI with:

```bash
python3 -m apps.cli.main
```

When first run, the CLI checks for a `config.json` file. If it does not exist, you will be prompted to enter URLs for the required services and your Datagen API key. The configuration keys are:

- `api_key`
- `Datagen backend`
- `Diffusers service`
- `Segmentators service`
- `Autodistill service`
- `ComfyUI`

The values are stored in `config.json` for subsequent runs.

### Workflow

1. **Dataset ID** – You will be asked to provide the target dataset ID. The CLI validates the ID using the Datagen API.
2. **Pipeline Selection** – Choose which pipeline to run. Currently available options are:
   - **Pipeline 1**
   - **Pipeline 2**
   - **Pipeline 3**
3. **CSV Input** – Provide the path to a CSV file that defines the conditioning data for the chosen pipeline.
4. The CLI iterates through each row, generates the image and corresponding annotations, then uploads them to the specified dataset.

## Repository Layout

- `apps/cli` – Implementation of the interactive command line interface.
- `core/pipelines` – Dataset generation pipelines used by the CLI.
- `datagen_sdk` – Minimal client for communicating with the Datagen API.
- `example_csv/` – Example CSV files demonstrating the expected column formats.

## Docker

A `Dockerfile` is provided for containerized execution. Build and run with:

```bash
docker build -t datagen-cli .
```

The container installs dependencies and executes **datagen-cli**.

