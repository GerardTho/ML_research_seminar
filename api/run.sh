#/bin/bash

uvicorn api.main:app --reload --host "0.0.0.0" --port 5000