FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from backend_files directory to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Start Flask app using Gunicorn
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:app"]
