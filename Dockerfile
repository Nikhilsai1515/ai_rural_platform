# Use Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the required port
EXPOSE 8080

# Command to run the app
CMD ["python", "main.py"]
