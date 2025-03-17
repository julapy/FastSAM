import io
import torch
import numpy as np
from PIL import Image
from fastsam import FastSAM
from pycocotools import mask as mask_utils

class EndpointHandler:
    def __init__(self, path=""):
        # Load model during initialization
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() 
            else "mps" if torch.backends.mps.is_available() 
            else "cpu"
        )
        print(f"Using device: {self.device}")
        
        # Path should point to the weights directory in the deployed model
        weights_path = f"{path}/weights/FastSAM-x.pt" if path else "./weights/FastSAM-x.pt"
        self.model = FastSAM(weights_path)
        print(f"Model loaded from {weights_path}")
        
        self.default_params = {
            "imgsz": 1024,
            "conf": 0.4,
            "iou": 0.9,
            "retina_masks": True
        }
    
    def mask_to_coco_rle(self, mask):
        """Encodes a mask into COCO RLE format."""
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        encoded["counts"] = encoded["counts"].decode("utf-8")  # Use ASCII format
        return encoded
    
    def __call__(self, data):
        """
        Args:
            data: Either the raw image data or a dictionary containing input data
        
        Returns:
            A list of dictionaries containing segmentation results
        """
        try:
            # Get image data with fallback pattern - if no "inputs" key, use the data itself
            image_data = data.pop("inputs", data) if isinstance(data, dict) else data
            
            # Process the image data
            if isinstance(image_data, str):
                # Base64 encoded image
                import base64
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            elif isinstance(image_data, bytes):
                # Raw bytes
                image = Image.open(io.BytesIO(image_data)).convert("RGB")
            else:
                return {"error": f"Invalid image format - Received type: {type(image_data)}"}
                    
            # Get parameters
            params = self.default_params.copy()
            if "parameters" in data and isinstance(data["parameters"], dict):
                # Update with user-provided parameters
                params.update(data["parameters"])
            
            # Run FastSAM model to generate masks
            results = self.model(
                image,
                self.device,
                retina_masks=params["retina_masks"],
                imgsz=params["imgsz"],
                conf=params["conf"],
                iou=params["iou"]
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
            segmentations = [self.mask_to_coco_rle(mask) for mask in masks]
            
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
            
            return response
            
        except Exception as e:
            # Log the error for debugging
            import traceback
            error_msg = str(e)
            error_traceback = traceback.format_exc()
            print(f"Error: {error_msg}")
            print(error_traceback)
            
            # Return error response
            return {"error": error_msg}