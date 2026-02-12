import json
import torch

from src.registries import MODELS

from alphazero.games.othello import OthelloNet

@MODELS.register("base")
def base_model(cfg):
    net = OthelloNet(n=cfg['board_size'])
    return net 

@MODELS.register("file")
def file_model(cfg):
    net = OthelloNet.from_pretrained(cfg['name'], cfg['checkpoint_path'])
    return net 


@MODELS.register("checkpoint_file")
def checkpoint_file_model(cfg):
    """
    Load a model directly from checkpoint weights + config JSON.

    Config:
        checkpoint_path: str
        config_path: str
    """
    with open(cfg["config_path"], "r") as f:
        json_config = json.load(f)

    model_config = OthelloNet.CONFIG(**json_config)
    net = OthelloNet(config=model_config)
    net.load_state_dict(torch.load(cfg["checkpoint_path"], map_location="cpu"))
    net.eval()
    return net
