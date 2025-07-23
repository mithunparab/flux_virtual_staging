#!/bin/bash

if [ ! -f .env ]; then
    echo ".env file not found!"
    exit 1
fi

export $(grep -v '^#' .env | xargs)

if [ -z "$GITHUB_USERNAME" ] || [ -z "$GITHUB_PAT" ] || [ -z "$REPO_NAME" ]; then
    echo "Missing environment variables. Please ensure GITHUB_USERNAME, GITHUB_PAT, and REPO_NAME are set."
    exit 1
fi

REPO_URL="https://${GITHUB_USERNAME}:${GITHUB_PAT}@github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

git config --global credential.helper cache

echo "Cloning repository..."
git clone "$REPO_URL"

git remote set-url "$REPO_NAME" "https://github.com/${GITHUB_USERNAME}/${REPO_NAME}.git"

echo "Clone complete. Remote URL sanitized."