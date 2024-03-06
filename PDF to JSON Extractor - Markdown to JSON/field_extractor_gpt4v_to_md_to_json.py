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
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Helpers import FileProcessingMetrics, Utilities

'''
    Field Extractor using GPT-4 Vision to extract markdown values from images in batches
    and GPT-4 Turbo to fill out the JSON fields using the markdown files.
    
    Inspired by Matt Groff:
        https://groff.dev/blog/ingesting-pdfs-with-gpt-vision
        https://github.com/mattlgroff/pdf-to-markdown
    
    The process is broken down into 4 steps:
    1) convert pdfs to images, (code)
    2) clean/pre-process images, (code)
    3) convert images to markdown, (GPT-4 Vision to interpret images and generate Markdown text.)
    4) retrieve JSON output from markdown (GPT-4 Turbo JSON Mode to fill out the JSON schema using the markdown files.)
        or generate a JSON schema from markdown if a schema is not provided.
'''

''' 
    EDIT THIS Configuration for the runner 
'''
PDF_FOLDER = "PDF Documents" # The folder containing the PDFs to be processed
USE_SCHEMA = True # Set to True if you have a JSON schema file to use for the JSON output
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
# GPT-4 Turbo 1106
GPT4T_1106_ENDPOINT = os.getenv("GPT4T-1106_ENDPOINT")
GPT4T_1106_API_KEY = os.getenv("GPT4T-1106_API_KEY")
GPT4T_1106_API_VERSION = os.getenv("GPT4T-1106_API_VERSION")
GPT4T_1106_CHAT_DEPLOYMENT_NAME = os.getenv("GPT4T-1106_CHAT_DEPLOYMENT_NAME")
GPT4T_1106_CHAT_MODEL = os.getenv("GPT4T-1106_CHAT_MODEL")

# Azure OpenAI client
client = AzureOpenAI(
    azure_deployment=GPT4T_1106_CHAT_DEPLOYMENT_NAME,
    azure_endpoint=GPT4T_1106_ENDPOINT,
    api_key=GPT4T_1106_API_KEY,
    api_version=GPT4T_1106_API_VERSION
)

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


def images_to_markdown(request_count, encoded_images):
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
                        "text": """
                            You are an expert at converting images to markdown.
                            Return the markdown text output from a single page originally from a scanned PDF.
                            Using formatting to match the structure of the page as close as you can get. 
                            Only output the markdown and nothing else. Do not explain the output, just return it. 
                            Do not use a single # for a heading. All headings will start with ## or ###. 
                            Skip over images and charts.
                            Return all text you see in markdown format.
                            Mark the page number at the top of the markdown's content for each corresponding image.
                            Do not explain your output or reasoning.
                        """
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """
                            In great detail, convert markdown text output from these images.
                            These images are from pages from a scanned PDF.
                        """
                    }
                ]
            }
        ],
        "temperature": 0,
        "top_p": 0,
        "max_tokens": 4096
    }

    # Add an item for each encoded image, limited to 10 images
    for encoded_image in encoded_images[:10]:
        payload["messages"][1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
        })

    return send_GPT4V_request_with_retry(request_count, headers, payload)


def process_images_to_markdown(image_folder="page_jpegs", markdown_folder="page_markdowns", final_markdown_file="final_markdown.md"):
    print("Processing images to markdown...")
    if os.path.exists(markdown_folder):
        shutil.rmtree(markdown_folder)
    if not os.path.exists(markdown_folder):
        os.makedirs(markdown_folder)

    # Get a list of all the images in the folder
    images = sorted(Path(image_folder).iterdir(), key=lambda x: x.stem)

    # ### TEST of specific range of images if running into issues
    # # Define the range of images to re-run and test
    # start_image_index = 43
    # end_image_index = 47

    # # Process the specified range of images to markdown
    # batch_images = images[start_image_index-1:end_image_index]
    # encoded_images = [encode_image_to_base64(str(image_path)) for image_path in batch_images]
    # markdown_content = images_to_markdown(encoded_images)

    # # Print the generated markdown content
    # print(markdown_content)
    # ### End of TEST

    ### Process images in batches
    # Adjust the batch size as needed
    batch_size = BATCH_SIZE
    # TODO **[Potentially] Look into how to adjust batch size based on a *token estimation* for the images in the batch in the process. 
    # NOTE     Need to not exceed 4096 completion tokens on the returning response from the model, as this means information is being cut off.
    request_count = 0
    for i in range(0, len(images), batch_size):

        # Process a batch of images to markdown
        request_count += 1
        batch_images = images[i:i+batch_size]
        batch_number = i // batch_size + 1
        total_batches = (len(images) + batch_size - 1) // batch_size
        num_images_in_batch = len(batch_images)
        print(f"Processing batch {batch_number} of {total_batches}. Batch size({batch_size} max): {num_images_in_batch} images.")
        encoded_images = [Utilities.encode_image_to_base64(str(image_path)) for image_path in batch_images]
        markdown_content = images_to_markdown(request_count, encoded_images)

        # Save the batch markdown content to a file
        start_page = i + 1
        end_page = min(i + batch_size, len(images))
        total_pages = len(images)
        digits = len(str(total_pages))
        output_filename = f"Pages_{str(start_page).zfill(digits)}_to_{str(end_page).zfill(digits)}.md"
        output_path = Path(markdown_folder) / output_filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
            print(f"Markdown for pages {start_page} to {end_page} saved to {output_path}")

    # Stick the markdown pages together
    stitch_markdown_pages(markdown_dir=markdown_folder, output_file=final_markdown_file)

    # clean markdown (GPT-4 Turbo to remove any irrelevant pieces like pretend images in the markdown, logos, pagent numbers, etc.)
    # #NOTE Omitting because it removes too much relevant information.
    # cleaned_markdown_directory = os.path.join(processing_output_folder, 'cleaned_page_markdowns')
    # process_markdown_files(input_directory_path=markdown_folder, output_directory_path=cleaned_markdown_directory)


def stitch_markdown_pages(markdown_dir, output_file):
    """Combine markdown files from a directory into a single markdown file."""
    files = os.listdir(markdown_dir)
    sorted_files = Utilities.sort_files_naturally(files)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in sorted_files:
            if filename.endswith('.md'):
                page_number = filename.split('_')[1].split('.')[0]
                with open(os.path.join(markdown_dir, filename), 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    outfile.write(f"# Page {page_number}\n\n{content}\n\n")
    
    print(f"Stitched markdown pages saved to: {output_file}")


'''
    NOTE Omitting because it removes too much relevant information.
    Optional: Have another pass of each Markdown file using GPT-4 Turbo to remove any irrelevant pieces like pretend images in the markdown, logos, pagent numbers, etc.
'''
def clean_markdown_content(text):
    """
    Sends the markdown text to OpenAI to remove irrelevant content.
    """
    start_time = time.time()
    response = client.chat.completions.create(
        model=GPT4T_1106_CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
                    You are tasked with cleaning up the following markdown text. 
                    You should return only the cleaned up markdown text. 
                    Do not explain your output or reasoning. 
                    Remove any irrelevant text from the markdown, returning the cleaned up version of the content. 
                    Examples include any images []() or 'click here' or 'Listen to this article' or page numbers or logos. 
                """
            },
            {
                "role": "user",
                "content": text,
            }
        ],
        temperature=0.1,
        max_tokens=4096
    )

    try:
        cleaned_content = response.choices[0].message.content if response.choices else ""
    except AttributeError:
        cleaned_content = "Error in processing markdown. Response format may have changed or is invalid."
        print(cleaned_content)

    elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
    print("--- Elapsed Time: %s ---" % elapsed_time)
    print(response.usage)
    FILE_PROCESSING_METRICS.update_token_usage(CURRENT_PDF_NAME, "GPT-4-Turbo 1106", "1", response.usage.prompt_tokens, response.usage.completion_tokens)
    
    return cleaned_content

def process_markdown_files(input_directory_path, output_directory_path):
    """
    Iterates through markdown files in the given input directory, cleans their content,
    and saves the cleaned content to a corresponding file in the output directory.
    """
    input_dir = Path(input_directory_path)
    output_dir = Path(output_directory_path)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"The directory {input_directory_path} does not exist.")
        return
    
    # Sort the files in alphanumeric order before processing
    sorted_files = sorted(input_dir.glob('*.md'), key=lambda path: path.stem)

    for markdown_file in sorted_files:
        print(f"Processing {markdown_file.name}...")
        with open(markdown_file, 'r', encoding='utf-8') as file:
            content = file.read()

        cleaned_content = clean_markdown_content(content)

        # Define the path for the cleaned file in the output directory
        cleaned_file_path = output_dir / markdown_file.name
        with open(cleaned_file_path, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
        print(f"Cleaned content saved to {cleaned_file_path}")


'''
    4. We leverage GPT-4 Turbo in JSON Mode to fill out the JSON fields using the markdown files.
    We prompt provide the model a JSON schema of the expected fields to search for as the output.
'''
def generate_json_from_markdown_template(final_markdown_file, json_output_file, schema_file=None):
    """
    Generates a JSON output from the given markdown file.
    It will either generate a schema and output or use a provided JSON schema.
    """
    print("Generating JSON output from the markdown file...")
    with open(final_markdown_file, 'r', encoding='utf-8') as file:
        markdown_content = file.read()

    start_time = time.time()
    messages = [
        {
            "role": "system",
            "content": f"""
                You are an expert in field extraction and image analysis. 
                Your task is to analyze the following markdown text and retrieve values for keys in a JSON object, using a provided JSON schema. 
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
        {
            "role": "user",
            "content": markdown_content,
        }
    ]

    if schema_file:
        with open(schema_file, 'r', encoding='utf-8') as file:
            schema = json.load(file)
            messages[0]["content"] += f" Fill out the fields using the provided JSON schema: {schema}"
    else:
        messages[0]["content"] += f"""
            Given a multi-page markdown file, generate a JSON schema that represents the structure and content of the markdown file. 
            The resulting JSON Object should include all important fields such as headers, subheaders, lists, links, and text blocks. 
            The importance of markdown elements should be determined based on their semantic significance in the document structure; For instance, headers and subheaders might be considered more important than regular text, and lists might be considered more important than individual list items. 
            Structure the JSON Object to be respective of each page, and sections.
        """
        # Please exclude any page numbers or page separations from the JSON object. 
    
    response = client.chat.completions.create(
        model=GPT4T_1106_CHAT_MODEL,
        temperature=0,
        response_format={ "type": "json_object" },
        messages=messages,
    )

    try:
        json_output = response.choices[0].message.content if response.choices else ""
    except AttributeError:
        json_output = "Error in processing JSON. Response format may have changed or is invalid."
        print(json_output)

    elapsed_time = datetime.timedelta(seconds=(time.time() - start_time))
    print("--- Elapsed Time: %s ---" % elapsed_time)
    print(response.usage)
    FILE_PROCESSING_METRICS.update_token_usage(CURRENT_PDF_NAME, "GPT-4-Turbo 1106", "1", response.usage.prompt_tokens, response.usage.completion_tokens)

    output_directory = os.path.dirname(json_output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    with open(json_output_file, 'w', encoding='utf-8') as file:
        json.dump(json.loads(json_output), file, indent=4, ensure_ascii=False)
        print(f"JSON output saved to {json_output_file}")


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
        markdown_files_directory = os.path.join(processing_output_folder, 'page_markdowns')
        final_markdown_path = os.path.join(processing_output_folder, pdf_name +'_markdown.md')
        json_files_directory = os.path.join(current_directory, 'json_outputs')
        json_output_path = os.path.join(json_files_directory, pdf_name + '_JSON.json')

        ### Process the PDF ###
        # 1) convert pdfs to images, (code)
        Utilities.pdf_to_images(pdf_path=pdf_path, dpi=IMAGE_CONVERTER_DPI, output_folder=image_files_directory)

        # 2) clean/pre-process images, (code)
        if PREPROCESS_IMAGES:
            Utilities.preprocess_images(image_folder=image_files_directory, output_folder=image_files_directory)

        # 3) images to markdown, (GPT-4 Vision to interpret and convert them into Markdown text.)
        process_images_to_markdown(image_folder=image_files_directory, markdown_folder=markdown_files_directory, final_markdown_file=final_markdown_path)

        # 4) convert markdown to JSON output (GPT-4 Turbo JSON Mode to fill out the JSON schema using the markdown files.)
        #    If a schema is not provided, generate a JSON schema from the markdown
        if USE_SCHEMA:
            generate_json_from_markdown_template(final_markdown_file=final_markdown_path, json_output_file=json_output_path, schema_file=schema_json_path)
        else:
            generate_json_from_markdown_template(final_markdown_file=final_markdown_path, json_output_file=json_output_path)
        
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
