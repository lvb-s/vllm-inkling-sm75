"""Ask the server about a generated test image (red circle, blue square, "CAT 42").

python3 image_chat.py http://127.0.0.1:8011   (needs Pillow; runs in the vLLM image)
"""

import base64
import io
import json
import sys
import urllib.request

from PIL import Image, ImageDraw, ImageFont

base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8011"
image = Image.new("RGB", (640, 400), "white")
draw = ImageDraw.Draw(image)
draw.ellipse((40, 60, 240, 260), fill=(220, 20, 20))
draw.rectangle((380, 60, 580, 260), fill=(20, 40, 220))
draw.text((200, 300), "CAT 42", fill="black", font=ImageFont.load_default(size=72))
buf = io.BytesIO()
image.save(buf, "PNG")
url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

body = {
    "model": "Inkling-2-Mini",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {
                    "type": "text",
                    "text": "Describe the shapes with their colors and positions, "
                    "and transcribe the text in the image.",
                },
            ],
        }
    ],
    "max_tokens": 1024,
    "temperature": 0,
}
req = urllib.request.Request(
    base + "/v1/chat/completions",
    json.dumps(body).encode(),
    {"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as resp:
    out = json.load(resp)
print("usage", out["usage"])
print(out["choices"][0]["message"]["content"])
