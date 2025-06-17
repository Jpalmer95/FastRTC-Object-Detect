# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
# This should be the main requirements.txt for the Gradio app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce layer size
# python:3.10-slim is Debian-based. opencv-python-headless generally has fewer system dependencies.
# We'll rely on pip to handle dependencies first. If specific .so errors occur during runtime,
# system libraries like libgl1 might be needed via apt-get.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY app.py .
COPY inference.py .
# If __pycache__ is not desired in the image, it can be excluded with a .dockerignore file,
# or by ensuring it's not copied if not tracked by git and copy uses git-tracked files.
# For this subtask, direct copy is fine.

# Make port 7860 available to the world outside this container (Gradio default)
# Cloud Run will use the PORT environment variable it sets, and Gradio's launch()
# (when server_port is not specified, as in fastRTC's Stream.run())
# will default to using the PORT env var if available, otherwise 7860.
EXPOSE 7860

# Run app.py when the container launches.
# Gradio should pick up the PORT environment variable set by Cloud Run.
CMD ["python", "app.py"]
