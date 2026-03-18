#!/bin/bash
mkdir -p data/uploads data/vector_store data/memory
uvicorn backend.main:app --host 0.0.0.0 --port $PORT