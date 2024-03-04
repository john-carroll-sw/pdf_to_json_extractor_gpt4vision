# GPT-4 Vision Field Extractor - Converts PDFs to JSON using schema

Large Language Models (LLMs) have the potential to transform the way we access and use data from legacy documents (like PDFs or Word files). They serve as a bridge connecting the past with the present, empowering organizations to fully leverage their historical records.

This repository contains implementations of a field extractor for taking PDF documents to output JSON using OpenAI's GPT-4 Vision Preview, offering flexibility to either generate a new JSON object or populate an existing schema. This functionality can be particularly useful for organizations looking to digitize their data, streamline their workflows, and unlock the full potential of their document archives.

#### *Note 
There are several Python libraries that can help you convert a PDF to JSON using text extraction, such as (for Python): PyPDF2, PDFMiner, Tabula-py. These libraries extract text from PDFs, so they work best with PDFs that contain selectable text. If your PDF contains **scanned image**s, you might need to use an OCR (Optical Character Recognition) library like pytesseract in combination with these libraries -- or use an LLM such as GPT-4 Vision.

## Implementation 1: Simple PDF to JSON

The first implementation is designed to handle scanned or digital PDFs of various documents, typically between 1 to 5 pages. It's a simpler implementation to show how to use GPT-4 Vision to extract form fields from a PDF and return a JSON Object structure representing these fields.

## Implementation 2: PDF > Images (batches) > JSON

This implementation can handle larger scanned or digital PDFs by sending up to 10 images per request to GPT-4 Vision. There is even retry logic for 429 errors. The process involves converting PDFs to images, cleaning and pre-processing these images, and then extracting JSON values from the images by checking for Key-Value pairs from the JSON Schema. This continues until all images have been processed. Finally it uses an ensemble method and combines all JSON from the batches for its final result. *Optimize for your use.

## Implementation 3: PDF > Images > Markdown > JSON

This implementation performs better for digital PDFs, instead of scanned. 
It also may provide a higher level of accuracy depending on your use case. It first converts the PDFs into images. It then uses GPT-4 Vision to convert these images into Markdown which is great for keeping the emphasis of the elements on the page. Also, GPT-4 Vision can interpret charts, images, on pages, which is something most OCR libraries cannot do. Finally, it converts the collective markdown into a JSON Object structure using GPT-4 Turbo's JSON Mode OR it can make a JSON Object schema from the markdown.
    
Inspired by Matt Groff's implementation:
* https://groff.dev/blog/ingesting-pdfs-with-gpt-vision
* https://github.com/mattlgroff/pdf-to-markdown
    

## Usage

To use these implementations, you will need to have Python installed and the necessary dependencies, which can be installed using pip, your own venv, or conda.

Pick the program you wish to experiment with to best serve your needs.

# Optimizations:

Here are some ways to optimize the OCR capabilities of GPT-4 Vision:

## 1) JSON Schema
* Supplying a JSON schema can improve consistency.

* Use a small, yet concise JSON schema. The larger the schema, the more fields to search for and fill, thus the model will take longer to return a response.

* Define the expected structure of the extracted information using the schema.

* GPT-4 Vision can then align its responses with the defined schema, leading to more deterministic results.

* The schema should cover field names, data types, and any constraints (e.g., required fields).

## 2) Prompt Engineering
* Use a more specific prompt to get the best results from the model.

* Suggest providing the model with a schema and prompt specific to the type of document being processed.
    i.e, if it's a document specific to a certain industry, country, language, jargon, 

* If it's a DMV document, prompt the model to extract the fields from a DMV, describe the layout, sections and fields, as well as how to best read the document.

* If it's a medical document, prompt the model explaining that it's a medical document, describe the sections and layout of the document.

## 3) Pre-Processing of Images

* Adjust the pre-processing of the images to optimize for OCR.

* Can adjust the pdf to image conversion DPI, too low and there won't be enough detail, too high and it's unnecessarily expensive to process with the LLM model.

* Can adjust the image pre-processing techniques to improve the quality of the text extracted from the images when it's processed by the model, 
    especially if the images are from a scanned pdf: i.e Remove Noise, Sharpen, Contrast, etc.

## 4) Image Batch Size

* Try to get the image batch size for GPT 4 Vision as close to 10 as possible without sacrificing the quality of the output. This ensures PDF's with large amounts of pages get processed as quickly as possible. 
    
* Max output tokens is 4096 tokens.

* Would need to create an image token estimation function to estimate the number of tokens for each image.

* Then use the token estimation to determine the batch size.

* If the token estimation is too high, then the batch size will need to be reduced.

* Also, depending on the system context, prompt and its settings (Temperature, Top P), 
    the input tokens will drastically affect the output tokens being returned from the model.


# GPT-4 Vision & GPT-4 Turbo

[GPT-4 Vision](https://writesonic.com/blog/gpt-4-vision) and [GPT-4 Turbo](https://medium.com/version-1/exploring-the-capabilities-of-gpt-4-turbo-d90d26df7174) are both powerful models developed by OpenAI, but they have different capabilities and use cases.

## GPT-4 Vision

GPT-4 Vision is a large multimodal model that can analyze images and provide textual responses to questions about them[^1]. It can extract a wealth of details about the image in text form[^2]. However, the data is not consistently formatted or, in other words, "unstructured"[^2]. So, while GPT-4 Vision can provide detailed descriptions of images, it does not inherently output structured JSON from images.

## GPT-4 Turbo

On the other hand, GPT-4 Turbo(and also 3.5 Turbo) has a feature called JSON mode, which ensures valid JSON output[^6][^8][^9]. This feature addresses previous challenges of generating JSON, such as improperly escaped characters, and facilitates data structuring[^6]. With the new JSON mode, developers can instruct GPT-4 Turbo to return structured data, vital for consistency and reliability in applications involving web development, data analytics, and machine learning[^6].

## Conclusion

So, if your primary requirement is to get structured JSON output, GPT-4 Turbo with its JSON mode would be a better choice. However, if you need to analyze images and get detailed descriptions, GPT-4 Vision would be more suitable. Please note that the exact details might vary based on the specific version of the API and the specific engine you're using. Always refer to the official [OpenAI API documentation](https://api.openai.com/v1/chat/completions) for the most accurate information.

If your main objective is to do both, then something you might do is allow GPT-4 Vision to convert the images to markdown, and then allow GPT-4 Turbo to convert this markdown to JSON.

> Source: Conversation with Bing, 2/23/2024

[^1]: [How to use the GPT-4 Turbo with Vision model - Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision)
[^2]: [Extracting Structured Data from Images Using OpenAI’s gpt-4 ... - Medium](https://medium.com/@foxmike/extracting-structured-data-from-images-using-openais-gpt-4-vision-and-jason-liu-s-instructor-ec7f54ee0a91)
[^3]: [Experimenting with GPT-4 Turbo’s JSON Mode: A New Era in AI ... - Medium](https://medium.com/@vishalkalia.er/experimenting-with-gpt-4-turbos-json-mode-a-new-era-in-ai-data-structuring-58d38409f1c7)
[^4]: [Unpacking GPT4-Turbo: Better in Every Way | Promptly Engineering](https://promptly.engineering/resources/gpt4-turbo)
[^5]: [Azure OpenAI Service Launches GPT-4 Turbo and GPT-3.5-Turbo-1106 Models](https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-openai-service-launches-gpt-4-turbo-and-gpt-3-5-turbo-1106/ba-p/3985962)
[^6]: [python - OpenAI API: How do I enable JSON mode using the gpt-4-vision](https://stackoverflow.com/questions/77434808/openai-api-how-do-i-enable-json-mode-using-the-gpt-4-vision-preview-model)
[^7]: [GPT-4 Vision - The Ultimate Guide](https://writesonic.com/blog/gpt-4-vision)
[^8]: [admineral/GPT4-Vision-React-Starter - GitHub](https://github.com/admineral/GPT4-Vision-React-Starter)
[^9]: [Exploring the Capabilities of GPT-4 Turbo - Medium](https://medium.com/version-1/exploring-the-capabilities-of-gpt-4-turbo-d90d26df7174)
[^10]: [undefined](https://api.openai.com/v1/chat/completions)
[^11]: [undefined](https://platform.openai.com/docs/models/gpt-4-and-gpt-4-turbo)

License
This project is licensed under the terms of the MIT license.
