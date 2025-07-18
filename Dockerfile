# Use a slim Python base image for efficiency
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install curl for health check
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy packaging configuration files first for better caching
COPY pyproject.toml .
COPY requirements.txt .

# Copy source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY portfolio_simulator_ui.py .

# Install the package in editable mode with dependencies
RUN pip install --no-cache-dir -e .

# Expose the default Streamlit port
EXPOSE 8501

# Health check (optional, but good practice)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit app using the legacy entry point for compatibility
CMD ["streamlit", "run", "portfolio_simulator_ui.py", "--server.port=8501", "--server.address=0.0.0.0"]