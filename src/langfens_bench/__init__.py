"""Inference-only RQ1 benchmark; configure deterministic/offline policy before torch."""
import os

os.environ["MKL_CBWR"] = "COMPATIBLE"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
