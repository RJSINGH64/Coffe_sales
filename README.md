# ☕ Coffee Sales Prediction App

**Developed by:** Rajat Singh  
**Internship:** Unified Mentor

---

## Overview

This README provides a step-by-step guide to deploy the Coffee Sales Prediction App, which predicts coffee sales prices using a Streamlit web application.

## Table of Contents

1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Setup Instructions](#setup-instructions)
4. [Development Steps](#development-steps)
5. [Deployment Steps](#deployment-steps)
6. [Usage](#usage)
7. [Exploratory Data Analysis](#exploratory-data-analysis)
8. [License](#license)

## Features

- Predict coffee sales prices based on user inputs.
- Visualize sales data with various plots.
- User-friendly interface using Streamlit.

## Technologies Used

- Python
- Streamlit
- Docker

## Setup Instructions

0. **Create Python environment 3.12 and Activate it**
   - For this Project 

1. **Clone the Repository**
   - Clone the project repository from GitHub to your local machine.

2. **Create a Virtual Environment (Optional)**
   - Set up a Python  environment 3.9 to manage dependencies. 

3. **Install Dependencies**
   - Use the provided `requirements.txt` to install necessary packages.

## Development Steps

1. **Create `app.py`**
   - Build the main application file using Streamlit.

2. **Prepare `requirements.txt`**
   - List all required packages for the project.

3. **Create `research.ipynb`**
   - Perform exploratory data analysis on the dataset.

## Deployment Steps

1. **Build the Docker Image**
   - Create a Dockerfile to define the application environment.

2. **Push the Docker Image to DockerHub**
   - Log in to DockerHub and push the built image for public access.

3. **Set Up AWS EC2**
   - Launch an EC2 instance on AWS with Docker installed.

4. **SSH into EC2 Instance**
   - Connect to your EC2 instance using SSH.

5. **Install Docker on EC2**
   - Ensure Docker is installed and running on your EC2 instance.

6. **Run Deployment Script**
   - Execute commands to pull the latest Docker image and run the application container.

7. **Access Your Streamlit App**
   - Open your web browser and navigate to the EC2 instance’s public IP at port `8501`.

## Usage

- Open the Streamlit app in your web browser.
- Use the provided interface to make predictions and visualize data.

## Exploratory Data Analysis

- The `research.ipynb` file contains the EDA performed on the coffee sales dataset.
- Open the notebook to view insights and visualizations.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

`
-**wsl commands**
`

# WSL and Docker Commands

## WSL (Windows Subsystem for Linux) Commands

### Starting WSL
- Start WSL (default distribution):
  ```bash
  wsl
  ```
- Start a specific distribution:
  ```bash
  wsl -d <DistroName>
  ```

### Managing Distributions
- List installed distributions:
  ```bash
  wsl --list --verbose
  ```
- Set a default distribution:
  ```bash
  wsl --set-default <DistroName>
  ```
- Set version for a specific distribution (WSL 1 or 2):
  ```bash
  wsl --set-version <DistroName> <Version>
  # Example: wsl --set-version Ubuntu 2
  ```

### Configuration and Initialization
- Open the `.bashrc` file for startup commands:
  ```bash
  nano ~/.bashrc
  ```
- Open the `.profile` file for startup commands:
  ```bash
  nano ~/.profile
  ```
- Check WSL version:
  ```bash
  wsl --list --verbose
  ```
- Set the default WSL version for new installations:
  ```bash
  wsl --set-default-version <Version>
  ```

### Managing Instances
- Terminate a specific distribution:
  ```bash
  wsl --terminate <DistroName>
  ```
- Shut down all WSL instances:
  ```bash
  wsl --shutdown
  ```

### Accessing Windows Files
- Access the C: drive:
  ```bash
  cd /mnt/c/
  ```

### WSL Configuration
- Check WSL configuration status:
  ```bash
  wsl --status
  ```

## Docker Commands

### Basic Docker Commands
- Check Docker version:
  ```bash
  docker --version
  ```

### Container Management
- List all containers (running and stopped):
  ```bash
  docker ps -a
  ```
- Start a specific container:
  ```bash
  docker start <ContainerID or ContainerName>
  ```
- Stop a specific container:
  ```bash
  docker stop <ContainerID or ContainerName>
  ```
- Remove a specific container:
  ```bash
  docker rm <ContainerID or ContainerName>
  ```

### Image Management
- List all images:
  ```bash
  docker images
  ```
- Pull an image from a registry:
  ```bash
  docker pull <ImageName>
  ```
- Remove a specific image:
  ```bash
  docker rmi <ImageID or ImageName>
  ```

### Pruning and Cleanup
- Remove stopped containers:
  ```bash
  docker container prune
  ```
- Remove unused images:
  ```bash
  docker image prune
  ```
- Remove unused volumes:
  ```bash
  docker volume prune
  ```
- Remove all unused resources (containers, networks, images, volumes):
  ```bash
  docker system prune
  ```
- Remove all unused images (including non-dangling ones):
  ```bash
  docker system prune -a
  ```

### Docker Compose (if applicable)
- Start services defined in a `docker-compose.yml` file:
  ```bash
  docker-compose up
  ```
- Stop services:
  ```bash
  docker-compose down
  ```

## Conclusion
This document provides a comprehensive list of commands for managing WSL and Docker. Modify or add additional commands as necessary for your specific use cases.


## EC2 aws commands for awscli

# Streamlit App Deployment on EC2 using Docker and GitHub Actions

This guide provides step-by-step commands to deploy a Streamlit app on an EC2 instance using Docker, with automatic deployment through GitHub Actions.

## Commands for EC2 Setup and Deployment

1. **Launch an EC2 Instance** using the AWS Management Console with your preferred settings.

2. **Connect to Your EC2 Instance**:
# Streamlit App Deployment on EC2 using Docker

This guide provides step-by-step commands to manually deploy a Streamlit app on an EC2 instance using Docker.

## Commands for EC2 Setup and Deployment

```bash
# 1. Launch an EC2 Instance using the AWS Management Console with your preferred settings.

# 2. Connect to Your EC2 Instance
ssh -i /path/to/your-key.pem ec2-user@your-ec2-public-dns

# 3. Install Docker (if not already installed)
sudo yum update -y
sudo amazon-linux-extras install docker
sudo service docker start
sudo usermod -a -G docker ec2-user
exit

# After executing the above commands, reconnect to your EC2 instance.

# 4. Install Docker Compose (optional)
sudo yum install -y python3-pip
sudo pip3 install docker-compose

# 5. Create a Directory for Your App
mkdir ~/my-streamlit-app
cd ~/my-streamlit-app

# 6. Create a Dockerfile
echo 'FROM python:3.8-slim

WORKDIR /app

# Copy requirements.txt first for better caching
COPY requirements.txt .

# Install dependencies without cache
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port on which your app will run
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "your_app.py", "--server.port=8501", "--server.address=0.0.0.0"]' > Dockerfile

# 7. Create requirements.txt
echo 'streamlit
pandas
numpy
# Add other dependencies' > requirements.txt

# 8. Build the Docker Image
docker build --no-cache -t my-streamlit-app .

# 9. Run the Docker Container
docker run -d --name my-streamlit-app -p 8501:8501 my-streamlit-app

# 10. Stop the Docker Container (if needed)
docker stop my-streamlit-app

# 11. Remove the Docker Container (if needed)
docker rm my-streamlit-app

# 12. Access Your Streamlit App
# You can access your Streamlit app by navigating to:
# http://your-ec2-public-dns:8501
