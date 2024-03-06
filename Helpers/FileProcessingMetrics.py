from Helpers.FileData import FileData
from Helpers.TokenUsage import TokenUsage

'''
    Keeps track of token usage for each file and model down to the request level
    Keeps track of the processing time and number of pages for each file
'''
class FileProcessingMetrics:
    def __init__(self):
        self.usage = {}

    def update_token_usage(self, file_path, model_name, request_number, prompt_tokens_used, completion_tokens_used):
        try:
            if file_path not in self.usage:
                self.usage[file_path] = {
                    "file_data": FileData(0, 0),
                    "models": {},
                }
            if model_name not in self.usage[file_path]["models"]:
                self.usage[file_path]["models"][model_name] = {}
            if f"request_{request_number}" not in self.usage[file_path]["models"][model_name]:
                self.usage[file_path]["models"][model_name][f"request_{request_number}"] = TokenUsage()

            self.usage[file_path]["models"][model_name][f"request_{request_number}"].add_prompt_tokens(prompt_tokens_used)
            self.usage[file_path]["models"][model_name][f"request_{request_number}"].add_completion_tokens(completion_tokens_used)

            # Update summary
            if "summary" not in self.usage:
                self.usage["summary"] = {}
            if model_name not in self.usage["summary"]:
                self.usage["summary"][model_name] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            self.usage["summary"][model_name]["prompt_tokens"] += prompt_tokens_used
            self.usage["summary"][model_name]["completion_tokens"] += completion_tokens_used
            self.usage["summary"][model_name]["total_tokens"] += prompt_tokens_used + completion_tokens_used

        except Exception as e:
            print(f"An error occurred while updating token usage: {e}")

    def print_global_token_usage(self):
        # Print total token usage for each model
        print("Total token usage for each model:")
        for model_name, summary in self.usage["summary"].items():
            print(f"    {model_name}:")
            print(f"        Prompt tokens: {summary['prompt_tokens']}")
            print(f"        Completion tokens: {summary['completion_tokens']}")
            print(f"        Total tokens: {summary['total_tokens']}")

    def update_file_data(self, file_path, processing_time, num_pages):
        try:
            if file_path not in self.usage:
                self.usage[file_path] = {
                    "file_data": FileData(),
                    "models": {}
                }
            self.usage[file_path]["file_data"].processing_time = processing_time
            self.usage[file_path]["file_data"].num_pages = num_pages

        except Exception as e:
            print(f"An error occurred while updating file data: {e}")

    def print_all_token_usage_for_each_file(self, file_path):
        # Print token usage for the file
        usage = self.usage[file_path]
        print(f"Token usage for {file_path}:")
        total_tokens = 0  # Initialize total tokens counter
        for model_name, requests in usage["models"].items():
            print(f"  Model: {model_name}")
            for request_number, tokens in requests.items():
                print(f"    Request {request_number}:")
                print(tokens)  # This will call the __str__ method of the TokenUsage class
                total_tokens += tokens.total_tokens  # Add total tokens for each model
        print(f"Total tokens for {file_path}: {total_tokens}\n")

    def print_total_token_usage_for_each_file(self, file_path):
        usage = self.usage.get(file_path)
        # print(f"File: {file_path}")
        # print(f"   - Processing time: {usage['file_data'].processing_time}")
        # print(f"   - Number of pages: {usage['file_data'].num_pages}")
        print(f"    Total token usage:")
        for model_name, requests in usage["models"].items():
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0  # Initialize total tokens counter
            print(f"      Model: {model_name}")
            for request_number, tokens in requests.items():
                total_prompt_tokens += tokens.prompt_tokens
                total_completion_tokens += tokens.completion_tokens
                total_tokens += tokens.total_tokens  # Add total tokens for each model
            print(f"          Total prompt tokens: {total_prompt_tokens}")
            print(f"          Total completion tokens: {total_completion_tokens}")
            print(f"          Total token usage: {total_tokens} tokens")

            print()

            # Calculate costs
            cost_library = {
                "GPT-4-Vision Preview": {
                    "prompt_token_cost": 0.01 / 1000,
                    "completion_token_cost": 0.03 / 1000
                },
                "GPT-4-Turbo": {
                    "prompt_token_cost": 0.01 / 1000,
                    "completion_token_cost": 0.03 / 1000
                }
            }

            prompt_token_cost = total_prompt_tokens * cost_library[model_name]["prompt_token_cost"]
            completion_token_cost = total_completion_tokens * cost_library[model_name]["completion_token_cost"]
            total_cost = prompt_token_cost + completion_token_cost

            print(f"          Prompt token cost: ${prompt_token_cost:.2f}")
            print(f"          Completion token cost: ${completion_token_cost:.2f}")
            print(f"          Total cost: ${total_cost:.2f}")
            print()

        print()

