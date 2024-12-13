#!/bin/bash

# Variables
PMPNN_S3_URL="https://absci-prod-ai-public-data.s3.us-west-2.amazonaws.com/igmpnn_acvr2b_holdout.ckpt"
LMD_S3_URL="https://absci-prod-ai-public-data.s3.us-west-2.amazonaws.com/igdesign_acvr2b_holdout.ckpt"
LOCAL_DIR="ckpts"
PMPNN_FILE_NAME="igmpnn_acvr2b_holdout.ckpt"
LMD_FILE_NAME="igdesign_acvr2b_holdout.ckpt"

# Create the local directory if it does not exist
if [ ! -d "$LOCAL_DIR" ]; then
  mkdir -p "$LOCAL_DIR"
fi

# Download the file using curl
curl -o "$LOCAL_DIR/$PMPNN_FILE_NAME" "$PMPNN_S3_URL"
curl -o "$LOCAL_DIR/$LMD_FILE_NAME" "$LMD_S3_URL"

# Verify the download
if [ -f "$LOCAL_DIR/$PMPNN_FILE_NAME" ]; then
  echo "File downloaded successfully to $LOCAL_DIR/$PMPNN_FILE_NAME"
else
  echo "Failed to download the file."
fi

if [ -f "$LOCAL_DIR/$LMD_FILE_NAME" ]; then
  echo "File downloaded successfully to $LOCAL_DIR/$LMD_FILE_NAME"
else
  echo "Failed to download the file."
fi
