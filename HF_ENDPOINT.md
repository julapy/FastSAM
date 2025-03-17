# FastSAM Hugging Face Inference Endpoint

This section describes how to use FastSAM as a custom Hugging Face Inference Endpoint.

## Using the Endpoint

### Basic Request

```python
import requests
import base64
from PIL import Image
import io

# Load image
image = Image.open("path/to/your/image.jpg")
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

# API endpoint
API_URL = "https://your-endpoint-url.huggingface.cloud"
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

# Request data
data = {
    "inputs": encoded_image,
    "parameters": {
        "imgsz": 1024,  # Image size
        "conf": 0.4,    # Confidence threshold
        "iou": 0.9,     # IoU threshold for filtering
        "retina_masks": True  # Generate high-quality masks
    }
}

# Send request
response = requests.post(API_URL, headers=headers, json=data)
result = response.json()

# Process results
# result is a list of dictionaries, each containing a segmentation mask and metadata
```

### Parameters

The endpoint accepts the following parameters:

- `imgsz`: Input image size (default: 1024)
- `conf`: Confidence threshold (default: 0.4)
- `iou`: IoU threshold for filtering (default: 0.9)
- `retina_masks`: Whether to generate high-quality masks (default: True)

### Response Format

The response is a list of dictionaries, where each dictionary represents a detected segment with the following fields:

- `segmentation`: Mask in COCO RLE format
- `area`: Area of the mask in pixels
- `bbox`: Bounding box in format [x, y, width, height]
- `predicted_iou`: Confidence score of the segmentation
- `point_coords`: Center point coordinates of the segment
- `stability_score`: Stability score (same as predicted_iou)
- `crop_box`: Cropping box used (same as bbox)

## Examples

### Python Example

```python
import requests
import json
import base64
from PIL import Image
import io
import numpy as np
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt

# Load and encode image
image_path = "examples/dogs.jpg"
image = Image.open(image_path)
buffered = io.BytesIO()
image.save(buffered, format="JPEG")
encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

# Make request
API_URL = "https://your-endpoint-url.huggingface.cloud"
API_TOKEN = "your_token_here"  # Replace with your token
headers = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

data = {
    "inputs": encoded_image,
    "parameters": {
        "imgsz": 1024,
        "conf": 0.4,
        "iou": 0.9,
        "retina_masks": True
    }
}

response = requests.post(API_URL, headers=headers, json=data)
result = response.json()

# Visualize results
plt.figure(figsize=(10, 10))
plt.imshow(image)

# Define colors for visualization
colors = [
    [255, 0, 0], [0, 255, 0], [0, 0, 255],
    [255, 255, 0], [255, 0, 255], [0, 255, 255],
    [128, 0, 0], [0, 128, 0], [0, 0, 128]
]

# Plot each segmentation mask with a different color
for i, segment in enumerate(result):
    # Convert RLE to binary mask
    rle = segment['segmentation']
    binary_mask = mask_utils.decode(rle)
    
    # Create a colored mask
    color_idx = i % len(colors)
    colored_mask = np.zeros((binary_mask.shape[0], binary_mask.shape[1], 4), dtype=np.uint8)
    colored_mask[binary_mask == 1] = colors[color_idx] + [128]  # Add alpha channel
    
    # Plot the mask
    plt.imshow(colored_mask, alpha=0.5)
    
    # Plot bounding box
    bbox = segment['bbox']
    plt.plot([bbox[0], bbox[0] + bbox[2], bbox[0] + bbox[2], bbox[0], bbox[0]],
             [bbox[1], bbox[1], bbox[1] + bbox[3], bbox[1] + bbox[3], bbox[1]],
             color=[c/255 for c in colors[color_idx]])

plt.axis('off')
plt.title(f'FastSAM Segmentation Results: {len(result)} segments')
plt.savefig("fastsam_results.png", bbox_inches='tight')
plt.show()
```

## Testing Locally

To test the endpoint locally before deployment:

```bash
python app.py --img_path ./images/dogs.jpg
```

## Deploying Your Own Endpoint

1. Fork this repository
2. Configure your Hugging Face Inference Endpoint with:
   - Framework: Custom
   - Docker image: your-username/fastsam-endpoint:latest
   - Handler class: handler:EndpointHandler
3. Deploy and enjoy real-time segmentation!