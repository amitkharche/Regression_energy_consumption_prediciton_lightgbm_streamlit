#!/bin/bash

# 🐳 Build Docker image
docker build -t energy-lightgbm-app .

# 🚀 Run Docker container
docker run -p 8501:8501 energy-lightgbm-app