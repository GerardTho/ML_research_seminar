"""A simple API to expose our trained RandomForest model for Tutanic survival."""
from fastapi import FastAPI, File, UploadFile
from typing import Annotated
from model import GNN_model
import torch
import pickle
import os
import json

import pandas as pd

model = GNN_model.GCN(num_node_features=3, num_classes=6, hidden_channels=16)
model.load_state_dict(torch.load('model/weights/ENZYMES_GCN_mean_None'))
model.eval()

app = FastAPI(
    title="blbllb",
    description="MARCHE"
    )


@app.get("/", tags=["Welcome"])
def show_welcome_page():
    """
    Show welcome page with model name and version.
    """

    return {
        "Message": "API",
        "Model_name": 'gcn',
        "Model_version": "0.1",
    }

@app.post('/best_model',tags=['Best model'])
async def bestmodel(
    mol_type: str
) -> dict :
    if mol_type not in ['ENZYMES','MUTAG','PROTEINS'] :
        raise ValueError('Wrong molecule type')
    
    results_path = 'model/results/'
    results_dicts = [os.path.join(results_path, i) for i in os.listdir(results_path) if os.path.isfile(os.path.join(results_path, i)) and mol_type in i]

    best_model = {}
    best_mean = 0

    for dico in results_dicts:
        f = open(dico)
        dico_result = json.load(f)
        if dico_result['mean_accuracy'] > best_mean:
            best_mean = dico_result['mean_accuracy']
            best_model = dico_result
    
    return best_model


@app.post("/predict", tags=["Predict"])
async def predict(
    file: UploadFile,
    mol_type: str
) -> int:

    if mol_type not in ['ENZYMES','MUTAG','PROTEINS'] :
        raise ValueError('Wrong molecule type')

    graph = pickle.load(file.file)
    out, loss = model(graph['x'], graph['edge_index'], None)
    return out.argmax(dim=1)