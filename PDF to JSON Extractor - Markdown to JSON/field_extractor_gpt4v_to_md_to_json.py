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


"""
    Optimizations:
        1) JSON Schema
            ○ Supplying a JSON schema for the crash report can improve consistency.
            ○ Use a small, yet concise JSON schema. The larger the schema, the more fields to search for and fill, thus the model will take longer to return a response.
            ○ Define the expected structure of the extracted information using the schema.
            ○ GPT-4 Vision can then align its responses with the defined schema, leading to more deterministic results.
            ○ The schema should cover field names, data types, and any constraints (e.g., required fields).
        2) Prompt Engineering
            ○ Use a more specific prompt to get the best results from the model.
            ○ Suggest providing the model with a schema and prompt specific to the type of document being processed.
                i.e, if it's a document specific to a certain industry, country, language, jargon, 
            ○ If it's a crash report document, prompt the model to extract the fields from a crash report, describe the layout, sections and fields, as well as how to best read the document.
            ○ If it's a lease document, prompt the model explaining that it's a lease document, describe the sections and layout of the document.
        3) Pre-Processing of Images
            ○ Adjust the pre-processing of the images to optimize for OCR.
            ○ Can adjust the pdf to image conversion DPI, too low and there won't be enough detail, too high and it's unnecessarily expensive to process with the LLM model.
            ○ Can adjust the image pre-processing techniques to improve the quality of the text extracted from the images when it's processed by the model, 
                especially if the images are from a scanned pdf: i.e Remove Noise, Sharpen, Contrast, etc.
        4) Image Batch Size
            ○ Try to get the image batch size for GPT 4 Vision as close to 10 as possible without sacrificing the quality of the output. 
                Max output tokens is 4096 tokens.
            ○ Would need to create an image token estimation function to estimate the number of tokens for each image.
            ○ Then use the token estimation to determine the batch size.
            ○ If the token estimation is too high, then the batch size will need to be reduced.
            ○ Also, depending on the system context, prompt and its settings (Temperature, Top P), 
                the input tokens will drastically affect the output tokens being returned from the model.
"""

# EDIT THIS Configuration for the runner
PDF_FOLDER = "PDF Documents" # The folder containing the PDFs to be processed
USE_SCHEMA = False # Set to True if you have a JSON schema file to use for the JSON output
JSON_SCHEMA_FILE = "document_schema.json" # The JSON schema file to use if USE_SCHEMA is True
IMAGE_CONVERTER_DPI = 200 # DPI
PREPROCESS_IMAGES = False # Set to True to enable image preprocessing for OCR optimization
BATCH_SIZE = 10 # The number of images to process in each batch. Max is 10 for GPT-4 Vision Preview

### Env Configuration
# Load environment variables from .env file
load_dotenv()
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
GLOBAL_TOKEN_USAGE = {} # Global token usage dictionary to keep track of the total usage of tokens
CURRENT_PDF_NAME = None # The name of the current PDF being processed

'''
    Inspired by Matt Groff:
        https://groff.dev/blog/ingesting-pdfs-with-gpt-vision
        https://github.com/mattlgroff/pdf-to-markdown
    
    1) convert pdfs to images, (code)
    2) clean/pre-process images, (code)
    3) convert images to markdown, (GPT-4 Vision to interpret images and generate Markdown text.)
    4) retrieve JSON output from markdown (GPT-4 Turbo JSON Mode to fill out the JSON schema using the markdown files.)
        or generate a JSON schema from markdown if a schema is not provided.
    
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


'''
    3. Once we have these images, we leverage OpenAI's GPT-4 Vision to interpret and convert them into Markdown text. 
    GPT-4 Vision's advanced capabilities allow it to understand complex layouts and visuals, ensuring that the converted 
    Markdown retains the richness and structure of the original PDF. It is prompted to retain the original layout to the best of it's ability. 
'''

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

    return send_request_with_retry(request_count, headers, payload)


def send_request_with_retry(request_id, headers, payload):
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


def process_images_to_markdown(image_folder="page_jpegs", markdown_folder="page_markdowns", final_markdown_file="converted-pdf.md"):
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
    # TODO **Adjust batch size based on a *token estimation* for the images in the batch in the markdown process. 
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
        encoded_images = [encode_image_to_base64(str(image_path)) for image_path in batch_images]
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
    sorted_files = sort_files_naturally(files)
    
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
    update_token_usage(CURRENT_PDF_NAME, "GPT-4-Turbo 1106", "1", response.usage.prompt_tokens, response.usage.completion_tokens)
    
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

def sort_files_naturally(files):
    """Sort the files in natural order to handle the numbering correctly."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(files, key=alphanum_key)


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
                You are a field extraction model. When given a series of documents, extract all the fields into a json structure.                
                You are tasked with filling out JSON fields using the following markdown text. 
                You should return JSON object as your output. 
                Do not explain your output or reasoning.
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
    update_token_usage(CURRENT_PDF_NAME, "GPT-4-Turbo 1106", "1", response.usage.prompt_tokens, response.usage.completion_tokens)

    output_directory = os.path.dirname(json_output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    with open(json_output_file, 'w', encoding='utf-8') as file:
        json.dump(json.loads(json_output), file, indent=4, ensure_ascii=False)
        print(f"JSON output saved to {json_output_file}")


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
        markdown_files_directory = os.path.join(processing_output_folder, 'page_markdowns')
        final_markdown_path = os.path.join(processing_output_folder, pdf_name +'_markdown.md')
        json_files_directory = os.path.join(current_directory, 'json_outputs')
        json_output_path = os.path.join(json_files_directory, pdf_name + '_JSON.json')

        ### Process the PDF ###
        # 1) convert pdfs to images, (code)
        pdf_to_images(pdf_path=pdf_path, dpi=IMAGE_CONVERTER_DPI, output_folder=image_files_directory)

        # 2) clean/pre-process images, (code)
        if PREPROCESS_IMAGES:
            preprocess_images(image_folder=image_files_directory, output_folder=image_files_directory)

        # 3) images to markdown, (GPT-4 Vision to interpret and convert them into Markdown text.)
        process_images_to_markdown(image_folder=image_files_directory, markdown_folder=markdown_files_directory, final_markdown_file=final_markdown_path)

        # 4) convert markdown to JSON output (GPT-4 Turbo JSON Mode to fill out the JSON schema using the markdown files.)
        #    If a schema is not provided, generate a JSON schema from the markdown
        if USE_SCHEMA:
            generate_json_from_markdown_template(final_markdown_file=final_markdown_path, json_output_file=json_output_path, schema_file=schema_json_path)
        else:
            generate_json_from_markdown_template(final_markdown_file=final_markdown_path, json_output_file=json_output_path)
        
        print(f"PDF {pdf_name} processing complete.")
        print_elapsed_time(pdf_start_time)
        print_token_usage_for_pdf(pdf_name)

        # Calculate the processing time for the current PDF, add it to the dictionary
        pdf_processing_time = time.time() - pdf_start_time
        pdf_processing_times[pdf_name] = pdf_processing_time
    
    # End of the process
    print(f"Extractor Runner ended at {datetime.datetime.now()}")
    print("Processing times per PDF:")
    for pdf_name, processing_time in pdf_processing_times.items():
        minutes = int(processing_time // 60)
        seconds = int(processing_time % 60)
        print(f"    {pdf_name}: {minutes} minutes {seconds} seconds")

    print_elapsed_time(global_start_time)  # Print the global time elapsed
    print_global_token_usage()  # Print the total token usage for each model


# Run Extractor
runner()