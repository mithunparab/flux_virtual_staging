# Virtual Staging Server with FLUX.1

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
    python -m venv venv
    source venv/bin/activate  
    ```

3. **Install requirements:**

    ```bash
    pip install uv
    uv pip install -r requirements.txt
    ```

## Running the Server

Start the application by running `main.py`:

```bash
python main.py
```

The server will be available locally at `http://127.0.0.1:8000`.

## Exposing Your Server for Remote Access

### Cloudflare Tunnel

1. **Install `cloudflared`:** Follow the official [Cloudflare Tunnels guide](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/install-and-setup/tunnel-guide/).

2. **Authenticate `cloudflared`:**

    ```bash
    cloudflared tunnel login
    ```

3. **Create a tunnel:** Give it a memorable name.

    ```bash
    cloudflared tunnel create virtual-stager-tunnel
    ```

    This will generate a credentials file.

4. **Run the tunnel:** Point it to your local service running on port 8000.

    ```bash
    cloudflared tunnel --url http://localhost:8000
    ```

`cloudflared` will now output a public `.trycloudflare.com` URL. This is your public endpoint! It's secure, stable, and ready to be shared.

## How to Use the API

The API has two main endpoints. Use the base URL provided by your server or tunneling service (e.g., `https://your-tunnel.trycloudflare.com`).

### `POST /stage_upload`

Use this for sending image files directly.

**Example `curl`:**

```bash
curl -X POST \
  -F "image=@/path/to/empty_room.jpg" \
  -F "prompt=A chic mid-century modern living room with a green velvet sofa" \
  -F "negative_prompt=ugly, blurry, bad lighting" \
  https://your-tunnel.trycloudflare.com/stage_upload \
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
  https://your-tunnel.trycloudflare.com/stage_base64 \
  -o staged_result_from_base64.png
```

### API Parameters

- `prompt` (string, required): Staging description. The system prompt is automatically added.
- `image` / `image_base64` (file/string, required): The input image.
- `negative_prompt` (string, optional): Describe what to avoid.
- `seed` (integer, optional): For reproducibility. If null/-1, a random seed is used.
- `guidance_scale` (float, optional): How strongly the prompt guides generation. Default: `2.5`.
- `steps` (integer, optional): Number of inference steps. Default: `28`.
