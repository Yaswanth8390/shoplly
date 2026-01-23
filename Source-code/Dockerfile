# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files to container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Hugging Face automatically maps this)
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
