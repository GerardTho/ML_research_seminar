"""A simple API to expose our trained RandomForest model for Tutanic survival."""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html
from model import GNN_model
import torch
from joblib import load
import pickle

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


@app.get("/predict", tags=["Predict"])
async def predict(
    index=0
) -> int:
    """
    """

    graph_file = open('data/datasets/ENZYMES/graphs/graph_'+str(index)+'.pickle', mode='rb')
    graph = pickle.load(graph_file)
    out, loss = model(graph['x'], graph['edge_index'], None)
    return out.argmax(dim=1)


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html(req: Request):
    root_path = req.scope.get("root_path", "").rstrip("/")
    openapi_url = root_path + app.openapi_url
    return get_swagger_ui_html(
        openapi_url=openapi_url,
        title="API",
    )