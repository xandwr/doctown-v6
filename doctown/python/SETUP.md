# Python Environment Setup

This guide explains how to set up and use the Python virtual environment for the Doctown embedding service.

## Prerequisites

- Python 3.8 or later
- pip (Python package installer)

## Local Development Setup

### Quick Setup

Run the setup script to automatically create and configure the virtual environment:

```bash
cd doctown/python
./setup_venv.sh
```

This script will:
1. Create a virtual environment in `doctown/python/venv/`
2. Upgrade pip to the latest version
3. Install all dependencies from `requirements.txt`

### Manual Setup

If you prefer to set up manually:

```bash
cd doctown/python

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Activating the Environment

After the initial setup, activate the virtual environment whenever you work on the project:

```bash
cd doctown/python
source venv/bin/activate
```

You'll see `(venv)` in your terminal prompt when the environment is active.

### Deactivating

When you're done working:

```bash
deactivate
```

## Running the Application

With the virtual environment activated:

```bash
# Run the main application
python app/main.py

# Run example scripts
python examples/test_embedder.py
```

## Docker Deployment

The Dockerfile automatically handles the Python environment setup:

1. Installs Python 3 and required system packages
2. Copies the Python application files
3. Creates a virtual environment inside the container
4. Installs all dependencies from `requirements.txt`

Build and run with Docker:

```bash
# From the doctown directory
docker build -t doctown .
docker run -p 8000:8000 doctown
```

The Python virtual environment in Docker is located at `/app/python/venv/` and is automatically used by the application.

## Managing Dependencies

### Adding New Dependencies

1. Activate the virtual environment
2. Install the package: `pip install <package-name>`
3. Update requirements.txt: `pip freeze > requirements.txt`

Or manually add to `requirements.txt` and run:

```bash
pip install -r requirements.txt
```

### Viewing Installed Packages

```bash
pip list
```

## Directory Structure

```
python/
├── venv/              # Virtual environment (not in git)
├── app/               # Main application code
├── config/            # Configuration files
├── examples/          # Example scripts
├── models/            # Downloaded models (not in git)
├── scripts/           # Utility scripts
├── requirements.txt   # Python dependencies
├── setup_venv.sh     # Setup script
├── .gitignore        # Git ignore rules
└── SETUP.md          # This file
```

## Troubleshooting

### Python version issues

Ensure you're using Python 3.8 or later:

```bash
python3 --version
```

### Permission errors

Make sure the setup script is executable:

```bash
chmod +x setup_venv.sh
```

### Module not found errors

Make sure:
1. The virtual environment is activated
2. Dependencies are installed: `pip install -r requirements.txt`

### Docker build issues

If the Docker build fails during Python setup:
- Check that `requirements.txt` is in the `python/` directory
- Ensure all dependencies are compatible with the Debian base image
- Check Docker build logs for specific error messages
