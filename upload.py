import argparse
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastsam import FastSAM 
import io
import torch
import traceback
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str, default="./weights/FastSAM-x.pt", help="model"
    )
    parser.add_argument(
        "--img_path", type=str, default="./images/dogs.jpg", help="path to image file"
    )
    parser.add_argument("--imgsz", type=int, default=1024, help="image size")
    parser.add_argument(
        "--iou",
        type=float,
        default=0.9,
        help="iou threshold for filtering the annotations",
    )
    parser.add_argument(
        "--text_prompt", type=str, default=None, help='use text prompt eg: "a dog"'
    )
    parser.add_argument(
        "--conf", type=float, default=0.4, help="object confidence threshold"
    )
    parser.add_argument(
        "--output", type=str, default="./output/", help="image save path"
    )
    parser.add_argument(
        "--randomcolor", type=bool, default=True, help="mask random color"
    )
    parser.add_argument(
        "--point_prompt", type=str, default="[[0,0]]", help="[[x1,y1],[x2,y2]]"
    )
    parser.add_argument(
        "--point_label",
        type=str,
        default="[0]",
        help="[1,0] 0:background, 1:foreground",
    )
    parser.add_argument("--box_prompt", type=str, default="[[0,0,0,0]]", help="[[x,y,w,h],[x2,y2,w2,h2]] support multiple boxes")
    parser.add_argument(
        "--better_quality",
        type=str,
        default=False,
        help="better quality using morphologyEx",
    )
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    parser.add_argument(
        "--device", type=str, default=device, help="cuda:[0,1,2,3,4] or cpu"
    )
    parser.add_argument(
        "--retina",
        type=bool,
        default=True,
        help="draw high-resolution segmentation masks",
    )
    parser.add_argument(
        "--withContours", type=bool, default=False, help="draw the edges of the masks"
    )
    return parser.parse_args()

app = FastAPI()

def mask_to_coco_rle(mask):
    """Encodes a mask into COCO RLE format."""
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    encoded["counts"] = encoded["counts"].decode("utf-8")  # Use ASCII format
    return encoded

@app.on_event("startup")
async def load_model():
    global model
    global device
    print("Loading model...")
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )    
    print(f"Using device: {device}")
    model = FastSAM("./weights/FastSAM-x.pt")
    print("Model loaded.")

@app.post("/segment/")
async def segment(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Run FastSAM model to generate masks
        results = model(
            image,
            device,
            retina_masks=True,
            imgsz=1024,
            conf=0.4,
            iou=0.9    
        )

        # Convert results to list
        results = list(results)

        # Extract masks, bounding boxes, and confidence scores
        masks = results[0].masks.data.cpu().numpy()  # Convert to NumPy
        bboxes = results[0].boxes.data.cpu().numpy().tolist()  # Bounding boxes
        predicted_ious = results[0].boxes.conf.cpu().numpy().tolist()  # Confidence scores
        areas = [int(np.sum(mask)) for mask in masks]  # Compute mask areas
        point_coords = [[[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]] for bbox in bboxes]  # Center points
        stability_scores = predicted_ious  # Use confidence as stability score
        crop_boxes = bboxes  # Use bbox as crop_box for now

        # Convert bbox and crop_box to integers
        bboxes = [[int(x) for x in bbox] for bbox in bboxes]
        crop_boxes = [[int(x) for x in crop_box] for crop_box in crop_boxes]

        # Encode masks in COCO RLE format
        segmentations = [mask_to_coco_rle(mask) for mask in masks]

        # Construct response matching SAM format
        response = [
            {
                "segmentation": segmentations[i],
                "area": areas[i],
                "bbox": bboxes[i],
                "predicted_iou": predicted_ious[i],
                "point_coords": point_coords[i],
                "stability_score": stability_scores[i],
                "crop_box": crop_boxes[i]
            }
            for i in range(len(segmentations))
        ]

        return JSONResponse(content=response)

    except Exception as e:
        print("Error:", e)
        traceback.print_exc()  # Print full error traceback to terminal
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)