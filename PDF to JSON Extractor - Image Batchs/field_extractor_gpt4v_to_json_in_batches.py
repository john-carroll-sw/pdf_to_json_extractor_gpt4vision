import os
import re
import shutil
import requests
import base64
import time
import json
import cv2
import datetime
import numpy as np
from math import ceil
from PIL import Image, ImageEnhance
from openai import AzureOpenAI
from pdf2image import convert_from_path
from pathlib import Path
from dotenv import load_dotenv


# EDIT THIS Configuration for the runner
PDF_FOLDER = "PDF Documents" # The folder containing the PDFs to be processed
JSON_SCHEMA_FILE = "document_schema.json" # JSON Schema used in the prompt
DOC_DESCRIPTION_PROMPT = """
    This document is a lease agreement between a landlord and a tenant.
    There will be multiple addresses in the document, but we are only interested in the address of the property being leased.
""" # Description of document to add context to the prompt to be used with the GPT-4 Vision API
IMAGE_CONVERTER_DPI = 200 # DPI
PREPROCESS_IMAGES = True # Set to True to enable image preprocessing for OCR optimization
BATCH_SIZE = 10 # The number of images to process in each batch. Max is 10 for GPT-4 Vision Preview

### Env Configuration
# Load environment variables from .env file
load_dotenv()
# GPT-4 Vision Preview
GPT4V_KEY = os.getenv("GPT4V_KEY")  # Your GPT4V key
GPT4V_ENDPOINT = os.getenv("GPT4V_ENDPOINT")  # The API endpoint for your GPT4V instance

# Logging Variables
GLOBAL_TOKEN_USAGE = {} # Global token usage dictionary to keep track of the total usage of tokens
CURRENT_PDF_NAME = None # The name of the current PDF being processed

class PdfData:
    def __init__(self, processing_time, num_pages):
        self.processing_time = processing_time
        self.num_pages = num_pages

'''
    1) convert pdfs to images, (code)
    2) clean/pre-process images, (code)
    3) extract json values from pdf's images, (GPT-4 Vision to check images for Key-Value pairs from the JSON Schema.)
'''

'''
    1. The process begins with the conversion of PDF documents into images, one for each page, using the pdf2image library. 
    This step is essential for capturing the entire content of the PDF, including charts and images that might be lost in simple text extractions.
'''
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
    2. Image Preprocessing for OCR Optimization: take from https://www.reveation.io/blog/automated-bank-statement-analysis/
    Image Preprocessing for OCR Optimization: The project aims to enhance OCR accuracy by implementing image preprocessing techniques. 
    These techniques include adaptive thresholding improve the quality of textual content within the images.
'''

def preprocess_image_for_ocr(image_path, output_path):
    print(f"Preprocessing image for OCR: {image_path}")
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's binarization
    # _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Gaussian adaptive thresholding
    # binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Convert NumPy array back to PIL Image
    enhanced_image = Image.fromarray(gray)

    # Enhance contrast
    enhancer = ImageEnhance.Contrast(enhanced_image)
    contrast_enhanced_image = enhancer.enhance(1.2)  # Experiment with enhancement factor

    # Save the pre-processed image as JPEG
    contrast_enhanced_image.save(image_path)


def preprocess_images(image_folder, output_folder):
    print(f"Processing images in the directory: {image_folder}")
    # Get a list of image files in the directory
    image_files = [file for file in os.listdir(image_folder) if file.endswith(".jpeg")]

    # Process each image in the directory
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        output_path = os.path.join(output_folder, image_file.split('.')[0] + "_processed.jpeg")
        # Call the functions to preprocess and encode the image
        preprocess_image_for_ocr(image_path, output_path)


# Please note that this is a simplification and the actual number of tokens may vary depending on the specific tokenization process used by the model.
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


def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

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
        print("The JSON string is not complete.")
    return json_object


def send_GPT4V_request_with_retry(request_id, headers, payload):
    max_retries = 3
    base_delay = 60  # seconds
    backoff_factor = 2 

    for retry in range(max_retries):
        try:
            start_time = time.time()
            response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
            json_response = response.json()
            break  # Break the loop if the request is successful
        except requests.RequestException as e:
            if response.status_code == 429:
                delay = base_delay * (backoff_factor ** retry) # exponential backoff
                print(f"Received 429 error. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise SystemExit(f"Failed to make the request. Error: {e}")

    elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
    print("--- Elapsed Time: %s ---" % elapsed_time)
    print(json_response["usage"])
    update_token_usage(CURRENT_PDF_NAME, "GPT-4-Vision Preview", str(request_id), json_response["usage"]["prompt_tokens"], json_response["usage"]["completion_tokens"])

    return json_response["choices"][0]["message"]["content"]


def extract_json_from_images(schema_file, image_folder="page_jpegs", json_output_folder="batch_json_outputs", final_json_file="final_output.json"):
    print("Extracting json from images...")
    if os.path.exists(json_output_folder):
        shutil.rmtree(json_output_folder)
    if not os.path.exists(json_output_folder):
        os.makedirs(json_output_folder)

    # Get a list of all the images in the folder
    images = sorted(Path(image_folder).iterdir(), key=lambda x: x.stem)

    ### Process images in batches
    # Adjust the batch size as needed
    batch_size = BATCH_SIZE
    # TODO **[Potentially] Look into how to adjust batch size based on a *token estimation* for the images in the batch in the process. 
    # NOTE     Need to not exceed 4096 completion tokens on the returning response from the model, as this means information is being cut off.
    request_count = 0
    for i in range(0, len(images), batch_size):

        # Process a batch of images
        request_count += 1
        batch_images = images[i:i+batch_size]
        batch_number = i // batch_size + 1
        total_batches = (len(images) + batch_size - 1) // batch_size
        num_images_in_batch = len(batch_images)
        print(f"Processing batch {batch_number} of {total_batches}. Batch size({batch_size} max): {num_images_in_batch} images.")
        encoded_images = [encode_image_to_base64(str(image_path)) for image_path in batch_images]
        response_content = query_for_schema_from_image_batch(request_count, schema_file, encoded_images)

        # Clean the response
        json_object = clean_json_response(response_content)

        # Save the batch content to a file
        start_page = i + 1
        end_page = min(i + batch_size, len(images))
        total_pages = len(images)
        digits = len(str(total_pages))
        output_filename = f"Pages_{str(start_page).zfill(digits)}_to_{str(end_page).zfill(digits)}.json"
        output_path = Path(json_output_folder) / output_filename
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(json_object, file, indent=4)
            print(f"JSON output from pages {start_page} to {end_page} saved to {output_path}")

    # Combine the JSON outputs into a single file
    combine_json_outputs(schema_file=schema_file, json_output_folder=json_output_folder, output_file=final_json_file)


def combine_json_outputs(schema_file, json_output_folder, output_file):
    """
        Combine all the JSON outputs into a single file.
        Combined data will include all the fields from the schema file, with values from the JSON outputs.
        It will only add the key-value pair to the combined data if the value is not None.
    """
    combined_data = {}

    # Load the schema file
    with open(schema_file, 'r') as schema:
        schema_data = json.load(schema)

    # Add fields from the schema file to the combined data
    combined_data.update({key: "" for key in schema_data})

    for filename in os.listdir(json_output_folder):
        if filename.endswith('.json'):
            with open(os.path.join(json_output_folder, filename), 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    if value:  # checks if the value is not None or not an empty string
                        combined_data[key] = value # Add the key-value pair to the combined data if the value is not None

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(combined_data, file, indent=4)
        print(f"All JSON files in the {json_output_folder} directory have been successfully combined into {output_file}.")


def sort_files_naturally(files):
    """Sort the files in natural order to handle the numbering correctly."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(files, key=alphanum_key)


def query_for_schema_from_image_batch(request_count, schema_file, encoded_images):
    if schema_file:
        with open(schema_file, 'r', encoding='utf-8') as file:
            schema = json.load(file)

    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_KEY,
    }
    payload = {
        "enhancements": {
            "ocr": {
                "enabled": True  # enabling OCR to extract text from the image using AI vision services
            },
            "grounding": {
                "enabled": True  # enabling grounding to extract the context of the image using AI vision services
            },
        },
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
                            You are an expert in field extraction and image analysis. 
                            Your task is to analyze images and retrieve values for keys in a JSON object, using a provided JSON schema. 
                            If an image contains a graphic, diagram, or chart, you should cease analysis of that image. 
                            Please note that explanations of your output or reasoning are not required.
                            Do not infer any data based on previous training, strictly use only source text given below as input.

                            Your JSON output should:
                            - be a JSON object that aligns with the provided schema. 
                            - remove all values for the keys not found in the image.
                            - always include all keys, even if the value is not found.

                            These images are originally from a document. Here are instructions related to the document:
                            {DOC_DESCRIPTION_PROMPT}
                        """
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Fill out the fields using the provided JSON schema: {schema}"
                    }
                ]
            }
        ],
        "temperature": 1, # Set to 0 to disable temperature sampling, default is 1
        "top_p": 1, # Set to 0 to disable nucleus sampling, default is 1
        "max_tokens": 1000
    }

    # Add an item for each encoded image, limited to 10 images
    for encoded_image in encoded_images[:10]:
        payload["messages"][1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
        })

    return send_GPT4V_request_with_retry(request_count, headers, payload)


def update_token_usage(pdf_path, model_name, request_number, prompt_tokens_used, completion_tokens_used):
    try:
        if pdf_path not in GLOBAL_TOKEN_USAGE:
            GLOBAL_TOKEN_USAGE[pdf_path] = {}
        if model_name not in GLOBAL_TOKEN_USAGE[pdf_path]:
            GLOBAL_TOKEN_USAGE[pdf_path][model_name] = {}
        if f"request_{request_number}" not in GLOBAL_TOKEN_USAGE[pdf_path][model_name]:
            GLOBAL_TOKEN_USAGE[pdf_path][model_name][f"request_{request_number}"] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

        GLOBAL_TOKEN_USAGE[pdf_path][model_name][f"request_{request_number}"]["prompt_tokens"] += prompt_tokens_used
        GLOBAL_TOKEN_USAGE[pdf_path][model_name][f"request_{request_number}"]["completion_tokens"] += completion_tokens_used
        GLOBAL_TOKEN_USAGE[pdf_path][model_name][f"request_{request_number}"]["total_tokens"] += prompt_tokens_used + completion_tokens_used

        # Update the model totals
        if "summary" not in GLOBAL_TOKEN_USAGE:
            GLOBAL_TOKEN_USAGE["summary"] = {}
        if model_name not in GLOBAL_TOKEN_USAGE["summary"]:
            GLOBAL_TOKEN_USAGE["summary"][model_name] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }

        GLOBAL_TOKEN_USAGE["summary"][model_name]["prompt_tokens"] += prompt_tokens_used
        GLOBAL_TOKEN_USAGE["summary"][model_name]["completion_tokens"] += completion_tokens_used
        GLOBAL_TOKEN_USAGE["summary"][model_name]["total_tokens"] += prompt_tokens_used + completion_tokens_used

    except Exception as e:
        print(f"An error occurred while updating token usage: {e}")


def print_token_usage_for_pdf(pdf_path):
    # Print token usage for the PDF
    usage = GLOBAL_TOKEN_USAGE[pdf_path]
    print(f"Token usage for {pdf_path}:")
    total_tokens = 0  # Initialize total tokens counter
    for model_name, requests in usage.items():
        print(f"  Model: {model_name}")
        for request_number, tokens in requests.items():
            print(f"    Request {request_number}:")
            print(f"      Prompt tokens: {tokens['prompt_tokens']}")
            print(f"      Completion tokens: {tokens['completion_tokens']}")
            print(f"      Total tokens: {tokens['total_tokens']}")
            total_tokens += tokens['total_tokens']  # Add total tokens for each model
    print(f"Total tokens for {pdf_path}: {total_tokens}\n")


def print_global_token_usage():
    # Print total token usage for each model
    print("\nTotal token usage for each model:")
    for model_name, usage in GLOBAL_TOKEN_USAGE["summary"].items():
        print(f"  {model_name}:\n"
        f"    Prompt tokens: {usage['prompt_tokens']}\n"
        f"    Completion tokens: {usage['completion_tokens']}\n"
        f"    Total tokens: {usage['total_tokens']}")


def print_elapsed_time(start_time):
    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    # Convert elapsed time to minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total elapsed time: {minutes} minutes {seconds} seconds")


def runner():
    global CURRENT_PDF_NAME, GLOBAL_TOKEN_USAGE
    print(f"Extractor Runner started at {datetime.datetime.now()}")
    global_start_time = time.time()
    pdf_processing_times = {} # Create an empty dictionary to store the processing times for each PDF
    current_directory = os.path.dirname(os.path.abspath(__file__)) # Configuration of paths

    # JSON Schema 
    schema_json_path = os.path.join(current_directory, JSON_SCHEMA_FILE) # Path to the JSON schema file if provided

    # PDF documents folder
    pdf_folder_name = PDF_FOLDER # Folder containing the PDFs to be processed

    # Run the extractor for every pdf in the folder
    pdf_folder_path = os.path.join(current_directory, pdf_folder_name)
    pdf_paths = [os.path.join(pdf_folder_path, file) for file in os.listdir(pdf_folder_path)]
    for pdf_path in pdf_paths:
        pdf_start_time = time.time()  # Start the timer for each pdf_path run

        ### Configuration ###
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        CURRENT_PDF_NAME = pdf_name
        print(f"Processing PDF: {pdf_name}")
        
        # Set-up processing output folders and files  
        processing_output_folder = os.path.join(current_directory, 'processing_outputs')
        image_files_directory = os.path.join(processing_output_folder, 'page_jpegs')
        json_files_directory = os.path.join(processing_output_folder, 'batch_json_outputs')
        json_output_path = os.path.join(current_directory, pdf_name + '_JSON.json')

        ### Process the PDF ###
        # 1) convert pdf to images, (code)
        pdf_to_images(pdf_path=pdf_path, dpi=IMAGE_CONVERTER_DPI, output_folder=image_files_directory)

        # 2) clean/pre-process images, (code)
        if PREPROCESS_IMAGES:
            preprocess_images(image_folder=image_files_directory, output_folder=image_files_directory)

        # 3) extract json values from pdf's images, (GPT-4 Vision to check images for Key-Value pairs from the JSON Schema.)
        extract_json_from_images(schema_file=schema_json_path, image_folder=image_files_directory, 
                                 json_output_folder=json_files_directory, final_json_file=json_output_path)

        print(f"PDF {pdf_name} processing complete.")
        print_elapsed_time(pdf_start_time)
        print_token_usage_for_pdf(pdf_name)

        # Calculate the processing time for the current PDF, add it to the dictionary
        pdf_processing_time = time.time() - pdf_start_time
        pdf_processing_times[pdf_name] = PdfData(
            pdf_processing_time,
            len([file for file in os.listdir(image_files_directory) if file.endswith(".jpeg")])
        )
    
    # End of the process
    print(f"Extractor Runner ended at {datetime.datetime.now()}")
    print("Processing times per PDF:")
    for pdf_name, pdf_data in pdf_processing_times.items():
        minutes = int(pdf_data.processing_time // 60)
        seconds = int(pdf_data.processing_time % 60)
        print(f"    {pdf_name}  ({pdf_data.num_pages} pages): {minutes} minutes {seconds} seconds")

    print_elapsed_time(global_start_time)  # Print the global time elapsed
    print_global_token_usage()  # Print the total token usage for each model


# Run Extractor
runner()
