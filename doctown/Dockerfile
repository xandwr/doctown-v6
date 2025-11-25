### Doctown Dockerfile ###
## This should cover everything for Doctown's builder pipeline in one, clean sweep. ##

# Stage 1: Build the Rust application
FROM rustlang/rust:nightly-slim as builder

# Install build dependencies for OpenSSL (required by reqwest)
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy Cargo files first for better caching
COPY rust/Cargo.toml rust/Cargo.lock* ./

# Create a dummy main.rs to build dependencies
RUN mkdir -p src && echo "fn main() {}" > src/main.rs

# Build dependencies (this layer will be cached)
RUN cargo build --release || true

# Remove the dummy source
RUN rm -rf src

# Copy the actual source code
COPY rust/src ./src

# Build the actual application
RUN cargo build --release

# Stage 2: Create the minimal runtime image
FROM debian:bookworm-slim

# Install runtime dependencies (SSL certificates, Python 3)
RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd -m -s /bin/bash doctown

WORKDIR /app

# Copy the built binary from the builder stage
COPY --from=builder /build/target/release/doctown /app/doctown

# Copy Python application
COPY python /app/python

# Set up Python virtual environment and install dependencies
RUN python3 -m venv /app/python/venv && \
    /app/python/venv/bin/pip install --upgrade pip && \
    /app/python/venv/bin/pip install -r /app/python/requirements.txt

# Change ownership to non-root user
RUN chown -R doctown:doctown /app

# Switch to non-root user
USER doctown

# Set the entrypoint
ENTRYPOINT ["/app/doctown"]

# Default arguments (can be overridden)
CMD ["--help"]