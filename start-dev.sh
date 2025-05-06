#!/bin/bash
set -e

# Function to initialize git repo if needed
initialize_git() {
  # Check if .git directory exists in the /app directory
  if [ ! -d "/app/.git" ]; then
    echo "Initializing git repository for development..."
    cd /app
    git init
    git config --global user.email "dev@example.com"
    git config --global user.name "Development Environment"
    git add .
    git commit -m "Initial development commit" || true
  else
    echo "Git repository already initialized."
  fi
}

# Initialize the backend data directory
initialize_backend() {
  # Create necessary directories if they don't exist
  mkdir -p /app/backend/data/cache/embedding/models
  mkdir -p /app/backend/data/cache/whisper/models
  mkdir -p /app/backend/data/cache/tiktoken

  # Ensure proper permissions
  chmod -R 755 /app/backend/data
}

# Setup npm projects if needed
setup_npm() {
  echo "Setting up Node.js environment..."
  cd /app

  # Check if node_modules exists
  if [ ! -d "/app/node_modules" ]; then
    echo "Installing npm dependencies..."
    rm -f package-lock.json || true
    npm install
  else
    echo "Node modules already installed."
  fi
}

# Main execution
echo "Starting OpenWebUI development environment..."

# Initialize git repository
initialize_git

# Initialize backend directories
initialize_backend

# Setup npm if needed
setup_npm

# Start the backend server
echo "Starting backend server..."
cd /app
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8080 --reload --reload-dir /app/backend