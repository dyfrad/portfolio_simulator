# Use a slim Python base image for efficiency
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY portfolio_simulator.py .

# Expose the default Streamlit port
EXPOSE 8501

# Health check (optional, but good practice)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the Streamlit app
CMD ["streamlit", "run", "portfolio_simulator.py", "--server.port=8501", "--server.address=0.0.0.0"]