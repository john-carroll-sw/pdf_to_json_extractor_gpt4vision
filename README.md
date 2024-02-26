# GPT-4 Vision Field Extractor - Converts PDFs to JSON using schema

This repository contains implementations of a field extractor for taking PDF documents to output JSON using OpenAI's GPT-4 Vision Preview.

## Implementation 1: Small Files

The first implementation is designed to handle smaller PDFs, typically between 1 to 5 pages. This implementation directly uses GPT-4 Vision to extract form fields from the PDF and returns a JSON Object structure representing these fields.

## Implementation 2: Large Files

The second implementation is designed to handle larger PDFs, such as those with 50 pages or more. This implementation first converts the PDFs into images. It then uses GPT-4 Vision to convert these images into Markdown. Finally, it converts the collective markdown into a JSON Object structure using GPT-4 Turbo's JSON Mode OR it can make a JSON Object schema from the markdown.

## Usage

To use these implementations, you will need to have Python installed and the necessary dependencies, which can be installed using pip, your own venv, or conda.

Pick the program you wish to experiment with to best serve your needs.


# GPT-4 Vision vs GPT-4 Turbo

[GPT-4 Vision](https://writesonic.com/blog/gpt-4-vision) and [GPT-4 Turbo](https://medium.com/version-1/exploring-the-capabilities-of-gpt-4-turbo-d90d26df7174) are both powerful models developed by OpenAI, but they have different capabilities and use cases.

## GPT-4 Vision

GPT-4 Vision is a large multimodal model that can analyze images and provide textual responses to questions about them[^1]. It can extract a wealth of details about the image in text form[^2]. However, the data is not consistently formatted or, in other words, "unstructured"[^2]. So, while GPT-4 Vision can provide detailed descriptions of images, it does not inherently output structured JSON from images.

## GPT-4 Turbo

On the other hand, GPT-4 Turbo has a feature called JSON mode, which ensures valid JSON output[^6][^8][^9]. This feature addresses previous challenges of generating JSON, such as improperly escaped characters, and facilitates data structuring[^6]. With the new JSON mode, developers can instruct GPT-4 Turbo to return structured data, vital for consistency and reliability in applications involving web development, data analytics, and machine learning[^6].

## Conclusion

So, if your primary requirement is to get structured JSON output, GPT-4 Turbo with its JSON mode would be a better choice. However, if you need to analyze images and get detailed descriptions, GPT-4 Vision would be more suitable. Please note that the exact details might vary based on the specific version of the API and the specific engine you're using. Always refer to the official [OpenAI API documentation](https://api.openai.com/v1/chat/completions) for the most accurate information.

If your main objective is to do both, then allow GPT-4 Vision to convert the images to markdown, and then allow GPT-4 Turbo to convert this markdown to JSON.

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
