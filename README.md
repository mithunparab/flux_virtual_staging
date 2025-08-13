# Virtual Staging

This application provides a robust server for virtual home staging. It exposes both a REST API and a web-based UI (Gradio), built on a FastAPI backend. It's designed for a single-GPU environment, using a queuing system to handle concurrent requests efficiently.

## Features

- **Smart Prompting**: Automatically instructs the model to preserve the room's background.
- **REST API**: Supports both file uploads and base64 JSON for flexible integration.
- **Web UI**: An easy-to-use interface for manual staging.
- **GPU Resource Management**: A job queue ensures that inference requests are processed one-by-one, preventing GPU OOM errors.
- **Performance**: Built with FastAPI and a background worker for low-latency request handling.

## Setup

1. **Clone the repository.**
2. **Create and activate a virtual environment.**

    ```bash
    python -m venv .venv
    source .venv/bin/activate  
    ```

3. **Install requirements:**

    ```bash
    cd $HOME && git clone https://github.com/black-forest-labs/flux
    cd flux
    pip install -e ".[tensorrt]" --extra-index-url https://pypi.nvidia.com
    ```

    ```bash
    pip install uv
    uv pip install -r requirements.lock
    ```

## Running the Server

**Step 1: Install `localtunnel`**

You need Node.js and `npm` installed on your server. Most Linux distributions have it. First, install `npm` if you don't have it:

```bash
sudo apt update && sudo apt install npm -y
```

Then, use `npm` to install `localtunnel` globally:

```bash
sudo npm install -g localtunnel
```

**Step 2: Run Your Python App**
In one SSH terminal, start your server as usual:

```bash
uvicorn main:app --host 0.0.0 --port 8000 --reload
```

Your app is now running on `localhost:8000`.

**Step 3: Start the Tunnel**

In a **second SSH terminal**, run this simple command:

```bash
echo "Password/Endpoint IP for localtunnel is: $(curl -s https://ipv4.icanhazip.com | tr -d '\n')"

lt --port 8000
```

The output will immediately give you your public URL:

```
Password/Endpoint IP for localtunnel is: 123.456.789.012
your url is: https://some-random-adjective-and-noun.loca.lt
```

## How to Use the API

The API has two main endpoints. Use the base URL provided by your server or tunneling service (e.g., `https://some-random-adjective-and-noun.loca.lt`).

### `POST /stage_upload`

Use this for sending image files directly.

**Example `curl`:**

```bash
curl -X POST \
  -F "image=@/path/to/empty_room.jpg" \
  -F "prompt=A chic mid-century modern living room with a green velvet sofa" \
  -F "negative_prompt=ugly, blurry, bad lighting" \
  https://some-random-adjective-and-noun.loca.lt/stage_upload \
  -o staged_room.png
```

### `POST /stage_base64`

Use this for sending a JSON payload with a base64-encoded image.

**Step 1: Encode image to base64**

```bash
base64 -w 0 empty_room.jpg > image.b64
```

**Step 2: Send `curl` request**

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "'$(cat image.b64)'",
    "prompt": "A luxurious art deco living room, with gold fixtures",
    "negative_prompt": "blurry, low quality, watermark",
    "seed": 123456
  }' \
  https://some-random-adjective-and-noun.loca.lt/stage_base64 \
  -o staged_result_from_base64.png
```

### API Parameters

- `prompt` (string, required): Staging description. The system prompt is automatically added.
- `image` / `image_base64` (file/string, required): The input image.
- `negative_prompt` (string, optional): Describe what to avoid.
- `seed` (integer, optional): For reproducibility. If null/-1, a random seed is used.
- `guidance_scale` (float, optional): How strongly the prompt guides generation. Default: `2.5`.
- `steps` (integer, optional): Number of inference steps. Default: `28`.
