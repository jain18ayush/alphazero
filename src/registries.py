# src/registries.py
from src.registry import Registry

DATASETS   = Registry("dataset")
CONCEPTS   = Registry("concept")
MODELS     = Registry("model")
HOOKS      = Registry("hook")
OPTIMIZERS = Registry("optimizer")
EVALUATORS = Registry("evaluator")
