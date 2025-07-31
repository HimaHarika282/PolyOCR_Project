import cv2
import pytesseract
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import numpy as np
from huggingface_hub import snapshot_download

local_path = "./trocr_handwritten"


# Loading TrOCR model
processor = TrOCRProcessor.from_pretrained(local_path)
model = VisionEncoderDecoderModel.from_pretrained(local_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function: Segmenting lines using OpenCV
def segment_lines(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Dilation to merge characters into lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 30))
    dilated = cv2.dilate(binary, kernel, iterations=1)

    # Finding contours for each line
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    line_images = []
    boxes = []

    for cnt in sorted(contours, key=lambda x: cv2.boundingRect(x)[1]):
        x, y, w, h = cv2.boundingRect(cnt)
        line_img = img[y:y+h, x:x+w]
        line_images.append(line_img)
        boxes.append((x, y, w, h))
    
    return line_images, boxes

# Function: Recognize text using TrOCR
def recognize_line_images(line_images):
    results = []
    for line in line_images:
        pil_img = Image.fromarray(line)
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        results.append(text)
    return results

# Main pipeline
def process_paragraph_image(image_path):
      image = cv2.imread(image_path)
      gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
      _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

      # Use dilation to merge characters into lines
      kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 5))  # width=40 joins nearby words
      dilated = cv2.dilate(binary, kernel, iterations=1)

      # Find contours after dilation
      contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
     
      lines = []
      for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 20:
            line_crop = image[y:y+h, x:x+w]
          if line_crop.ndim == 2:
              line_crop = cv2.cvtColor(line_crop, cv2.COLOR_GRAY2RGB)
          elif line_crop.shape[2] == 1:
              line_crop = cv2.merge([line_crop]*3)

          line_pil = Image.fromarray(line_crop)
          lines.append(((x, y, w, h), line_pil))

    # Sort lines top-to-bottom, then left-to-right
      lines_sorted = sorted(lines, key=lambda b: (b[0][1], b[0][0]))  

    # Run OCR on sorted lines
      for i, (box, line_image) in enumerate(lines_sorted):
        pixel_values = processor(images=line_image, return_tensors="pt").pixel_values.to(device)
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"Line {i+1}: {text}")
        print(f"Bounding Box: x={box[0]}, y={box[1]}, w={box[2]}, h={box[3]}")
        print("")
    


image_path = "h2.jpeg"
process_paragraph_image(image_path)
