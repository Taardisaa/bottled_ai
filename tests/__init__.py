import os


# Keep baseline handler tests deterministic unless explicitly overridden.
os.environ.setdefault("LLM_ENABLED", "false")
