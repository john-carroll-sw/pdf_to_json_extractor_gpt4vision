import base64
import json
import os
import re
import shutil
import time
from PIL import Image, ImageEnhance
from math import ceil
import cv2
from pdf2image import convert_from_path


class Utilities:
    '''
        The conversion of PDF documents into images, one for each page, using the pdf2image library. 
        This step is essential for capturing the entire content of the PDF, including charts and images that might be lost in simple text extractions.
    '''
    @staticmethod
    def pdf_to_images(pdf_path, dpi=300, output_folder="page_jpegs"):
        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            
        print(f"Converting PDF to images with DPI={dpi}...")
        images = convert_from_path(pdf_path, dpi=dpi, fmt='jpeg')
        total_pages = len(images)
        digits = len(str(total_pages))

        for i, image in enumerate(images):
            image_path = os.path.join(output_folder, f"Page_{str(i+1).zfill(digits)}.jpeg")
            image.save(image_path, "JPEG")
            print(f"Page {i+1} saved as image: {image_path}")


    '''
        Image Preprocessing for OCR Optimization: 
            The project aims to enhance OCR accuracy by implementing image preprocessing techniques. 
            These techniques include adaptive thresholding improve the quality of textual content within the images.
            Inspired from: https://www.reveation.io/blog/automated-bank-statement-analysis/
    '''
    @staticmethod
    def preprocess_image_for_ocr(image_path):
        print(f"Preprocessing image for OCR: {image_path}")
        # Load the image
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Otsu's binarization
        # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Gaussian adaptive thresholding
        # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Auto-rotate using pytesseract (https://pypi.org/project/pytesseract/)
        # img = cv2.imread(image)
        # k = pytesseract.image_to_osd(img)
        # out = {i.split(":")[0]: float_convertor(i.split(":")[-1].strip()) for i in k.rstrip().split("\n")}
        # img_rotated = ndimage.rotate(img, 360-out["Rotate"])

        # Convert NumPy array back to PIL Image
        enhanced_image = Image.fromarray(gray)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(enhanced_image)
        contrast_enhanced_image = enhancer.enhance(1.2)  # Experiment with enhancement factor

        # Save the pre-processed image as JPEG
        contrast_enhanced_image.save(image_path)

    @staticmethod
    def preprocess_images(image_folder, output_folder):
        print(f"Processing images in the directory: {image_folder}")
        # Get a list of image files in the directory
        image_files = [file for file in os.listdir(image_folder) if file.endswith(".jpeg")]

        # Process each image in the directory
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            output_path = os.path.join(output_folder, image_file.split('.')[0] + "_processed.jpeg")
            # Call the functions to preprocess and encode the image
            Utilities.preprocess_image_for_ocr(image_path, output_path)

    @staticmethod
    def clean_json_response(response_content):
        # Clean up the response's content. Convert the response's json string to a json object
        # Remove the leading and trailing characters
        json_string = response_content.replace('```json\n', '')
        json_string = json_string.rsplit('\n', 1)[0]
        try:
            # Try to parse the JSON string into a Python dictionary
            json_object = json.loads(json_string)
            # print(json_object)
        except json.JSONDecodeError:
            # NOTE: This may happen because GPT-4 Vision is still in preview and the JSON response may not be complete.
            print("The JSON string is not complete.")
        return json_object

    @staticmethod
    def count_image_tokens(image_path):
        # Open the image file
        with Image.open(image_path) as img:
            # Get image size
            width, height = img.size

        # Calculate the number of tokens based on the formula provided by OpenAI
        h = ceil(height / 512)
        w = ceil(width / 512)
        n = w * h
        total = 85 + 170 * n

        return total

    @staticmethod
    def encode_image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def sort_files_naturally(files):
        """Sort the files in natural order to handle the numbering correctly."""
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(files, key=alphanum_key)
    
    @staticmethod
    def print_elapsed_time(start_time):
        end_time = time.time()  # Stop the timer
        elapsed_time = end_time - start_time  # Calculate the elapsed time

        # Convert elapsed time to minutes and seconds
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        print(f"Total elapsed time: {minutes} minutes {seconds} seconds")

    @staticmethod
    def float_convertor(x):
        if x.isdigit():
            out= float(x)
        else:
            out= x
        return out 
