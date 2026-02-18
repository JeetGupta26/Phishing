#!/bin/bash
set -e

# Production Deployment Script
IMAGE_NAME="phishing-service"
CONTAINER_NAME="phishing-app"
PORT=8000

echo "🚀 Starting Deployment..."

# 1. Pull latest code (if not using pre-built image)
# git pull origin production

# 2. Build Image
echo "🛠️ Building Docker Image..."
docker build -t $IMAGE_NAME .

# 3. Stop and Remove old container
if [ "$(docker ps -aq -f name=$CONTAINER_NAME)" ]; then
    echo "🛑 Stopping existing container..."
    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME
fi

# 4. Run new container
echo "🚢 Starting new container..."
docker run -d \
    --name $CONTAINER_NAME \
    --restart always \
    -p 127.0.0.1:$PORT:8000 \
    --env-file .env \
    $IMAGE_NAME

# 5. Health Check
echo "🔍 Verifying Health..."
sleep 5
HEALTH=$(curl -s http://localhost:$PORT/health | grep -o '"ready":true')

if [ "$HEALTH" == '"ready":true' ]; then
    echo "✅ Deployment Successful!"
else
    echo "❌ Health Check Failed. Rolling back..."
    # Simple rollback logic could go here
    exit 1
fi
