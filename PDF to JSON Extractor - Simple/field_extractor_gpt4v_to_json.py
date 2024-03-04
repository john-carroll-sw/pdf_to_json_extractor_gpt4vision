from pdf2image import convert_from_path
import os
import requests
import base64
import time
import shutil
import json

# Configuration
GPT4V_KEY = os.getenv("GPT4V_KEY")  # Your GPT4V key
GPT4V_ENDPOINT = os.getenv("GPT4V_ENDPOINT")  # The API endpoint for your GPT4V instance

def convert_pdf_to_images(pdf_path, image_path):
    # Read the PDF and convert to images
    images = convert_from_path(pdf_path)
    for i, image in enumerate(images):
        fname = "page_" + str(i+1) + ".jpg"
        image.save(os.path.join(image_path, fname), "JPEG")


def encode_images(image_paths):
    encoded_images = []
    for image_path in image_paths:
        encoded_image = base64.b64encode(open(image_path, "rb").read()).decode("ascii")
        encoded_images.append(encoded_image)
    return encoded_images


def send_request(encoded_images):
    headers = {
        "Content-Type": "application/json",
        "api-key": GPT4V_KEY,
    }

    # Payload for the request
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
                            You are a field extraction expert. When given a series of images, extract all the fields into a JSON object structure.
                            Treat the series of documents as one cohesive document and return a json mapping all the appropriate fields.
                            Structure the JSON Object to be respective of each page, and sections.
                            In the JSON Object, supply information about the page using the Header.
                            You should return JSON object as your output. 
                            Do not explain your output or reasoning.
                        """
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Return the fields in this document as a complete json object",
                    },
                ],
            },
        ],
        "temperature": 0,
        "top_p": 0,
        "max_tokens": 4096,
    }

    # Add an item for each encoded image, limited to 10 images
    for encoded_image in encoded_images[:10]:
        payload["messages"][1]["content"].append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
        })

    # Send request
    try:
        start_time = time.time()
        response = requests.post(GPT4V_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
        json_response = response.json()
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")

    # Handle the response as needed (e.g., print or process)
    # time taken to get response
    print("--- %s seconds ---" % (time.time() - start_time))

    return json_response


def clean_json_response(response_content):
    # Clean up the response's content. Convert the response's json string to a json object
    # Remove the leading and trailing characters
    json_string = response_content.replace('```json\n', '')
    json_string = json_string.rsplit('\n', 1)[0]
    try:
        # Try to parse the JSON string into a Python dictionary
        json_object = json.loads(json_string)
        print(json_object)
    except json.JSONDecodeError:
        print("The JSON string is not complete.")
    return json_object


def write_json_to_file(pdf_path, json_object):
    # Create the 'JSON Output' folder if it doesn't exist
    output_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "JSON Output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Write the response to a JSON file in the 'JSON Output' folder
    output_filename = os.path.splitext(os.path.basename(pdf_path))[0] + ".json"
    output_filepath = os.path.join(output_folder, output_filename)
    with open(output_filepath, "w") as file:
        json.dump(json_object, file, indent=4)


def runner():
    start_time = time.time()  # Start the timer

    # PDF documents folder
    pdf_folder_name = "Reports"
    current_directory = os.path.dirname(os.path.abspath(__file__))
    pdf_folder_path = os.path.join(current_directory, pdf_folder_name)
    pdf_paths = [os.path.join(pdf_folder_path, file) for file in os.listdir(pdf_folder_path)]

    # Run the extractor for every pdf in the folder
    for pdf_path in pdf_paths:

        # Convert PDF's to Images
        image_folder_name = "Images" # Path to images
        image_path = os.path.join(current_directory, image_folder_name)

        # Clear the images folder
        shutil.rmtree(image_path)
        if not os.path.exists(image_path):
            os.makedirs(image_path)

        convert_pdf_to_images(pdf_path, image_path)

        # Get a list of all paths of the images in the image folder
        image_paths = [os.path.join(image_path, file) for file in os.listdir(image_path)]

        encoded_images = encode_images(image_paths)

        # Send Request to GPT4V to Return JSON structure from Images
        json_response = send_request(encoded_images)

        # Get the content from the response
        response_content = json_response["choices"][0]["message"]["content"]

        # Clean the response
        json_object = clean_json_response(response_content)

        # Output to Output folder directory
        write_json_to_file(pdf_path, json_object)

        # Print the response
        print(response_content)
        print("\n \n")

        # Print the tokens used
        print(json_response["usage"])


    end_time = time.time()  # Stop the timer
    elapsed_time = end_time - start_time  # Calculate the elapsed time

    # Convert elapsed time to minutes and seconds
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Elapsed time: {minutes} minutes {seconds} seconds")
    

# Run Extractor
runner()
