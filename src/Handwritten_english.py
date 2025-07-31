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
