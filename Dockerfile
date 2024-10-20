# Base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.enableCORS=false"]
