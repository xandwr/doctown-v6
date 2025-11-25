# Doctown Info

## What this is

`doctown/` is the folder containing EVERYTHING needed for deployment of the xandwrp/doctown:latest Docker image.
This is the entire pipeline condensed into a single, easy to deploy container.

Just the way God intended.

## What it should do

Right now, it needs to be stupid simple. Just input a github repo URL, and if it's public/accessible/valid,
we pull a clone of it locally, unzip it, and return a json of its folder structure repr.
Then delete local files and clean up and shut down automatically.

That's literally it. Once that works, everything else falls into place almost automatically.

## Mental model type shit

The doctown folder is the entire v6 brain that runs inside a single pod:
- Rust → deterministic CPU analysis
- Python → GPU embeddings
- Orchestrator → glues both together
- FastAPI → exposes a clean HTTP endpoint

Everything outside of this folder (website, CLI, etc.) is optional UI layered on top.

## Python Development Setup

The Python embedding service is located in `python/` with its own virtual environment.

**Quick start:**
```bash
cd python
./setup_venv.sh
source venv/bin/activate
```

For detailed setup instructions, see [python/SETUP.md](python/SETUP.md).

The Docker build automatically handles the Python environment setup - no manual steps needed for deployment.