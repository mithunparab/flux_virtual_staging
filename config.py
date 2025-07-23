import numpy as np

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
MAX_IMAGE_SIZE = 1024


MAX_SEED = np.iinfo(np.int32).max
DEFAULT_GUIDANCE_SCALE = 2.5
DEFAULT_STEPS = 28

SYSTEM_PROMPT = "Maintain the background wall, floor, and windows of the room. Only add furniture and objects into the foreground. "

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, unrealistic, bad lighting, watermark, text, signature, deformed, ugly, disfigured, distorted"

API_HOST = "0.0.0.0"
API_PORT = 8000
API_TIMEOUT = 240