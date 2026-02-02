FROM python:3.13-slim

WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy project files
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/

# Create data and logs directories
RUN mkdir -p data logs

# Install dependencies
RUN uv sync --no-dev

# Run the trading system
CMD ["uv", "run", "python", "scripts/run_live.py"]
