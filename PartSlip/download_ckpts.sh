#!/bin/bash

# Create a folder called models
mkdir -p models

# Download a model from a link to the created folder
wget -P models https://huggingface.co/datasets/minghua/PartSLIP/resolve/main/models/glip_large_model.pth