"""Pytest configuration and fixtures."""

import os

# Set API key so test modules can import inter_sim_rl.google_maps_api
_env_key = os.environ.get("API_KEY") or ""
if not _env_key.strip() or _env_key in ("YourAPIKey", "your_api_key_here"):
    os.environ["API_KEY"] = "pytest-env-key"
