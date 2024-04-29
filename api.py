"""A simple API to expose our trained RandomForest model for Tutanic survival."""
from fastapi import FastAPI, File, UploadFile
from typing import Annotated
from model import GNN_model
from model.layer_selector import local_pooling_selection, global_pooling_selection, conv_selection
import torch
import pickle
import os
import json

import pandas as pd

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


def find_bestmodel(
    mol_type: str
) -> dict:
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

@app.post('/best_model', tags=['Best model'])
async def bestmodel(
    mol_type: str
) -> dict:
    return find_bestmodel(mol_type)

dico_molecules = {
    'ENZYMES': {'Nb_class': 6, 'Nb_features': 3},
    'MUTAG': {'Nb_class': 6, 'Nb_features': 3},
    'PROTEINS': {'Nb_class': 2, 'Nb_features': 3},
}

@app.post("/predict", tags=["Predict"])
async def predict(
    file: UploadFile,
    mol_type: str
) -> int:

    if mol_type not in ['ENZYMES','MUTAG','PROTEINS'] :
        raise ValueError('Wrong molecule type')

    best_model_config = find_bestmodel(mol_type)
    graph = pickle.load(file.file)

    conv_method = conv_selection(best_model_config['convolution_layer'], best_model_config['attention_heads'])
    global_pool_method = global_pooling_selection(best_model_config['global_pooling_layer'])
    local_pool_method, dic_conversion_layer = local_pooling_selection(best_model_config['local_pooling_layer'], device='cpu')

    model = GNN_model.GCN(
        num_node_features=dico_molecules[mol_type]['Nb_features'],
        num_classes=dico_molecules[mol_type]['Nb_class'],
        hidden_channels=best_model_config['hidden_channels'],
        conv_method=conv_method,
        global_pool_method=global_pool_method,
        local_pool_method=local_pool_method,
        dic_conversion_layer=dic_conversion_layer
        )

    model.load_state_dict(torch.load(
        f"model/weights/{mol_type}_{best_model_config['convolution_layer']}_{best_model_config['global_pooling_layer']}_{best_model_config['local_pooling_layer']}"
        ))
    model.eval()

    out, loss = model(graph['x'], graph['edge_index'], None)
    return out.argmax(dim=1)