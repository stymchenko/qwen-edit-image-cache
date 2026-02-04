#!/bin/bash

# Build and push the cached model worker image to Docker Hub.
# Usage: ./build-and-push.sh <docker-hub-username>

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ -z "$1" ]; then
    echo -e "${RED}Error: Docker Hub username required${NC}"
    echo "Usage: ./build-and-push.sh <docker-hub-username>"
    exit 1
fi

DOCKER_USERNAME=$1
IMAGE_NAME="cached-model-worker"
IMAGE_TAG="${2:-latest}"
FULL_IMAGE="${DOCKER_USERNAME}/${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${GREEN}=== Building cached model worker ===${NC}"
echo "Image: ${FULL_IMAGE}"
echo ""

# Verify required files exist
for f in handler.py Dockerfile requirements.txt; do
    if [ ! -f "$f" ]; then
        echo -e "${RED}Error: ${f} not found${NC}"
        exit 1
    fi
done
echo -e "${GREEN}All required files found${NC}"
echo ""

# Build
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t "${FULL_IMAGE}" .
echo -e "${GREEN}Build successful${NC}"
echo ""

# Push
echo -e "${YELLOW}Pushing to Docker Hub...${NC}"
docker push "${FULL_IMAGE}"
echo -e "${GREEN}Push successful${NC}"
echo ""

echo -e "${GREEN}=== Done ===${NC}"
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""
echo "Next steps:"
echo "  1. Go to https://www.console.runpod.io/serverless"
echo "  2. Create a new endpoint with container image: ${FULL_IMAGE}"
echo "  3. Under Model, add: microsoft/Phi-3-mini-4k-instruct"
echo "  4. Deploy"
