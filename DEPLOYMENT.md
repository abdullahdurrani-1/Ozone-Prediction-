# 🚀 Deployment Guide

Complete guide to deploy the Ozone Prediction API to various platforms.

## Table of Contents
- [Local Development](#local-development)
- [Docker & Docker Compose](#docker--docker-compose)
- [AWS EC2](#aws-ec2)
- [Google Cloud Run](#google-cloud-run)
- [Heroku](#heroku)
- [DigitalOcean](#digitalocean)
- [Kubernetes](#kubernetes)

---

## Local Development

### Setup

```bash
# Clone and navigate
git clone https://github.com/yourname/ozone-api.git
cd ozone-api

# Setup (macOS/Linux)
bash setup.sh

# Or manual setup (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env with your Google API key
```

### Run

```bash
# Development server with auto-reload
python app.py

# Or with uvicorn
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Visit: http://localhost:8000/docs
```

### Test

```bash
python test_api.py
```

---

## Docker & Docker Compose

### Build Image

```bash
docker build -t ozone-api:latest .
```

### Run Single Container

```bash
docker run -d \
  --name ozone-api \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/models:/app/models \
  ozone-api:latest
```

### Using Docker Compose (Recommended)

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f ozone-api

# Stop services
docker-compose down

# Scale to multiple instances
docker-compose up -d --scale ozone-api=3
```

### Push to Docker Hub

```bash
# Login
docker login

# Tag image
docker tag ozone-api:latest yourusername/ozone-api:latest

# Push
docker push yourusername/ozone-api:latest
```

---

## AWS EC2

### Step 1: Launch Instance

```bash
# Use Ubuntu 22.04 LTS
# t3.medium or larger (2GB RAM minimum)
# Security group: allow 80, 443, 8000
```

### Step 2: Connect & Setup

```bash
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Logout and login again
exit
ssh -i your-key.pem ubuntu@your-instance-ip
```

### Step 3: Deploy

```bash
# Clone repo
git clone https://github.com/yourname/ozone-api.git
cd ozone-api

# Setup environment
nano .env  # Add your Google API key

# Copy models
scp -i your-key.pem -r models/* ubuntu@your-instance-ip:~/ozone-api/models/

# Start with Docker Compose
docker-compose up -d
```

### Step 4: Reverse Proxy (Nginx)

```bash
sudo apt-get install -y nginx

# Create config
sudo nano /etc/nginx/sites-available/default
```

Add this configuration:

```nginx
upstream ozone_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://ozone_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://ozone_api;
        access_log off;
    }
}
```

```bash
# Reload nginx
sudo nginx -s reload
```

### Step 5: SSL with Let's Encrypt

```bash
sudo apt-get install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

### Monitoring with CloudWatch

```bash
# Install CloudWatch agent
wget https://s3.amazonaws.com/amazoncloudwatch-agent/ubuntu/amd64/latest/amazon-cloudwatch-agent.deb
sudo dpkg -i amazon-cloudwatch-agent.deb
```

---

## Google Cloud Run

### Prerequisites

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash

# Initialize
gcloud init

# Set project
gcloud config set project YOUR_PROJECT_ID
```

### Deploy

```bash
# Build and push
gcloud run deploy ozone-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 3600 \
  --set-env-vars GOOGLE_API_KEY=your_key_here \
  --allow-unauthenticated
```

### Or using Cloud Build

```bash
# Create cloudbuild.yaml
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/ozone-api:latest
```

### Set up Cloud Storage for Models

```bash
# Create bucket
gsutil mb gs://YOUR_BUCKET_NAME

# Upload models
gsutil -m cp models/* gs://YOUR_BUCKET_NAME/

# Mount in Cloud Run
# Add in Dockerfile:
RUN gsutil cp -r gs://YOUR_BUCKET_NAME/models .
```

---

## Heroku

### Prerequisites

```bash
# Install Heroku CLI
brew tap heroku/brew && brew install heroku

# Login
heroku login
```

### Setup

```bash
# Create app
heroku create ozone-api

# Add buildpacks
heroku buildpacks:add heroku/python

# Set environment variables
heroku config:set GOOGLE_API_KEY=your_key_here
```

### Create Procfile

```
# Procfile
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

### Deploy

```bash
# Initialize git (if not already)
git init
git add .
git commit -m "Initial commit"

# Deploy
git push heroku main

# View logs
heroku logs --tail

# Add models to Heroku storage
# Note: Heroku has ephemeral storage, use S3 or similar
```

---

## DigitalOcean

### Step 1: Create Droplet

- Ubuntu 22.04 LTS
- 2GB RAM / 2 vCPU minimum
- Enable IPv6

### Step 2: Initial Setup

```bash
# SSH into droplet
ssh root@your_droplet_ip

# Update
apt-get update && apt-get upgrade -y

# Create non-root user
adduser appuser
usermod -aG sudo appuser

# Switch user
su - appuser
```

### Step 3: Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

### Step 4: Deploy

```bash
# Clone repo
git clone https://github.com/yourname/ozone-api.git
cd ozone-api

# Configure
nano .env

# Start container
docker-compose up -d
```

### Step 5: Setup App Platform (Optional)

```bash
# Use DigitalOcean App Platform for easy management
# Push to GitHub
git push origin main

# Connect GitHub repo in DigitalOcean dashboard
# Configure environment variables
# Deploy!
```

---

## Kubernetes

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Install Helm (optional)
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```

### Create Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ozone-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ozone-api
  template:
    metadata:
      labels:
        app: ozone-api
    spec:
      containers:
      - name: ozone-api
        image: yourusername/ozone-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: google-api-key
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: ozone-api-service
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: ozone-api
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ozone-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ozone-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Deploy to Kubernetes

```bash
# Create secret
kubectl create secret generic api-secrets --from-literal=google-api-key=your_key_here

# Apply deployment
kubectl apply -f deployment.yaml

# Check status
kubectl get deployments
kubectl get pods
kubectl get svc

# View logs
kubectl logs -f deployment/ozone-api

# Forward port locally
kubectl port-forward svc/ozone-api-service 8000:80
```

---

## Environment Variables

Key variables to set on all platforms:

```bash
GOOGLE_API_KEY=your_key_here
DEBUG=False
HOST=0.0.0.0
PORT=8000
ML_MODELS_PATH=/app/models/
DL_MODELS_PATH=/app/models/
SCALER_PATH=/app/models/scaler.pkl
```

---

## Performance Tuning

### For High Traffic

```bash
# Increase workers
uvicorn app:app --workers 4 --loop uvloop

# Or in docker-compose
services:
  ozone-api:
    command: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
```

### For Low Latency

- Deploy on GPU instances (for LSTM/GRU models)
- Use GRU instead of LSTM (faster inference)
- Cache predictions with Redis
- Use CDN for static content

### Monitoring & Logging

```bash
# Prometheus metrics (optional)
pip install prometheus-client

# ELK Stack for logging
docker run -d -p 9200:9200 docker.elastic.co/elasticsearch/elasticsearch:8.0.0
```

---

## Troubleshooting

### Models Not Loading

```bash
# Check permissions
chmod -R 755 models/

# Verify files exist
ls -la models/
```

### Out of Memory

```bash
# Check memory usage
free -h

# Increase swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### High CPU Usage

```bash
# Profile the app
py-spy record -o profile.svg -p PID

# Check which endpoint is slow
# Use time curl http://localhost:8000/endpoint
```

---

## Production Checklist

- [ ] Set up HTTPS/SSL
- [ ] Configure firewall rules
- [ ] Set up monitoring & alerts
- [ ] Configure backups
- [ ] Set up CI/CD pipeline
- [ ] Test auto-scaling
- [ ] Configure CDN
- [ ] Set up rate limiting
- [ ] Enable logging
- [ ] Document recovery procedures

---

For more help: Check [README.md](README.md) or open an issue!
