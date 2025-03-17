import argparse
import base64
import json
from typing import Dict, Any
import sys
import os
from PIL import Image
import io

from handler import EndpointHandler

def parse_args():
    parser = argparse.ArgumentParser(description="Test FastSAM endpoint locally")
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou", type=float, default=0.9, help="IoU threshold for filtering"
    )
    parser.add_argument(
        "--retina_masks", type=bool, default=True, help="Use retina masks"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize the handler
    handler = EndpointHandler()
    
    # Load and encode the image
    with open(args.img_path, "rb") as f:
        image_bytes = f.read()
    
    # Encode image to base64 for testing both input formats
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    
    # Test with raw bytes
    print("Testing with raw bytes input...")
    result_bytes = handler({"inputs": image_bytes})
    print(f"Found {len(result_bytes)} segments")
    
    # Test with base64
    print("Testing with base64 input...")
    result_base64 = handler({
        "inputs": base64_image,
        "parameters": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "retina_masks": args.retina_masks
        }
    })
    print(f"Found {len(result_base64)} segments")
    
    # Save results to a JSON file
    with open("test_result.json", "w") as f:
        json.dump(result_base64, f, indent=2)
    
    print(f"Results saved to test_result.json")

if __name__ == "__main__":
    main()