# src/registries.py
from src.registry import Registry

DATASETS   = Registry("dataset")
CONCEPTS   = Registry("concept")
COUNTERFACTUALS = Registry("counterfactuals")
MODELS     = Registry("model")
HOOKS      = Registry("hook")
OPTIMIZERS = Registry("optimizer")
EVALUATORS = Registry("evaluator")
