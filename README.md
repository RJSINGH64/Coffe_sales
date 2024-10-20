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

