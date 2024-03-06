import os
import shutil
import requests
import time
import json
import datetime
from pathlib import Path
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Helpers import FileProcessingMetrics, Utilities

'''
    Field Extractor using GPT-4 Vision to extract JSON values from images in batches
    
    The process is broken down into 3 steps:
    1) convert pdfs to images, (code)
    2) clean/pre-process images, (code)
    3) extract json values from pdf's images, (GPT-4 Vision to check images for Key-Value pairs from the JSON Schema.)
'''

''' 
    EDIT THIS Configuration for the runner 
'''
PDF_FOLDER = "PDF Documents" # The folder containing the PDFs to be processed
JSON_SCHEMA_FILE = "document_schema.json" # The JSON schema file to use if USE_SCHEMA is True
DOC_DESCRIPTION_PROMPT = """
    Be as precise as possible when extracting the information from the images.
""" # Description of document to add context to the prompt to be used with the GPT-4 Vision API
IMAGE_CONVERTER_DPI = 200 # DPI
PREPROCESS_IMAGES = True # Set to True to enable image preprocessing for OCR optimization
BATCH_SIZE = 10 # The number of images to process in each batch. Max is 10 for GPT-4 Vision Preview


### Env Configuration
load_dotenv() # Load environment variables from .env file
# GPT-4 Vision Preview
GPT4V_KEY = os.getenv("GPT4V_KEY")  # Your GPT4V key
GPT4V_ENDPOINT = os.getenv("GPT4V_ENDPOINT")  # The API endpoint for your GPT4V instance

# Logging Variables
CURRENT_PDF_NAME = None # The name of the current PDF being processed
FILE_PROCESSING_METRICS = FileProcessingMetrics() # File processing metrics for all runs, tracks tokens and processing times

'''
    Functions
'''
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
    FILE_PROCESSING_METRICS.update_token_usage(CURRENT_PDF_NAME, "GPT-4-Vision Preview", str(request_id), json_response["usage"]["prompt_tokens"], json_response["usage"]["completion_tokens"])

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
        encoded_images = [Utilities.encode_image_to_base64(str(image_path)) for image_path in batch_images]
        response_content = query_for_schema_from_image_batch(request_count, schema_file, encoded_images)

        # Clean the response
        json_object = Utilities.clean_json_response(response_content)

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

    files = os.listdir(json_output_folder)
    sorted_files = Utilities.sort_files_naturally(files)

    for filename in sorted_files:
        if filename.endswith('.json'):
            with open(os.path.join(json_output_folder, filename), 'r') as f:
                data = json.load(f)
                for key, value in data.items():
                    if value:  # checks if the value is not None or not an empty string
                        combined_data[key] = value # Add the key-value pair to the combined data if the value is not None

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(combined_data, file, indent=4)
        print(f"All JSON files in the {json_output_folder} directory have been successfully combined into {output_file}.")


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
        "temperature": 0, # Set to 0 to disable temperature sampling, default is 1
        "top_p": 0, # Set to 0 to disable nucleus sampling, default is 1
        "max_tokens": 4096
    }

    # Add an item for each encoded image, limited to 10 images
    for encoded_image in encoded_images[:10]:
        payload["messages"][1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
        })

    return send_GPT4V_request_with_retry(request_count, headers, payload)


def runner():
    global CURRENT_PDF_NAME, FILE_PROCESSING_METRICS
    print(f"Extractor Runner started at {datetime.datetime.now()}")
    global_start_time = time.time()
    current_directory = os.path.dirname(os.path.abspath(__file__)) # Configuration of paths
    schema_json_path = os.path.join(current_directory, JSON_SCHEMA_FILE) # Path to the JSON schema file if provided
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
        Utilities.pdf_to_images(pdf_path=pdf_path, dpi=IMAGE_CONVERTER_DPI, output_folder=image_files_directory)

        # 2) clean/pre-process images, (code)
        if PREPROCESS_IMAGES:
            Utilities.preprocess_images(image_folder=image_files_directory, output_folder=image_files_directory)

        # 3) extract json values from pdf's images, (GPT-4 Vision to check images for Key-Value pairs from the JSON Schema.)
        extract_json_from_images(schema_file=schema_json_path, image_folder=image_files_directory, 
                                 json_output_folder=json_files_directory, final_json_file=json_output_path)

        # Print the processing times for the PDF and the total token usage for each model
        print(f"PDF {pdf_name} processing complete.")
        Utilities.print_elapsed_time(pdf_start_time)
        FILE_PROCESSING_METRICS.print_all_token_usage_for_each_file(pdf_name)

        # Update the processing times for the PDF and the number of pages
        pdf_processing_time = time.time() - pdf_start_time
        FILE_PROCESSING_METRICS.update_file_data(pdf_name, pdf_processing_time, 
            len([file for file in os.listdir(image_files_directory) if file.endswith(".jpeg")]))
    
    # End of the process
    print("========================================================")
    print(f"Extractor Runner ended at {datetime.datetime.now()}")
    print("\nProcessing times for each PDF and the total token usage for each model:")
    for file_path, file_data in FILE_PROCESSING_METRICS.usage.items():
        if file_path != 'summary':
            print(f"  {file_data['file_data'].get_formatted_processing_time(file_path)}")
            FILE_PROCESSING_METRICS.print_total_token_usage_for_each_file(file_path)

    # Print the global time elapsed and the global total token usage for each model
    Utilities.print_elapsed_time(global_start_time)  # Print the global time elapsed
    FILE_PROCESSING_METRICS.print_global_token_usage()  # Print the global total token usage for each model


# Run Extractor
runner()
