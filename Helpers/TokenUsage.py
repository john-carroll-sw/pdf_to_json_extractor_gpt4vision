class TokenUsage:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def add_prompt_tokens(self, count):
        self.prompt_tokens += count
        self.calculate_total_tokens()

    def add_completion_tokens(self, count):
        self.completion_tokens += count
        self.calculate_total_tokens()

    def calculate_total_tokens(self):
        self.total_tokens = self.prompt_tokens + self.completion_tokens

    def __str__(self):
        return f"      Prompt tokens: {self.prompt_tokens}\n" \
               f"      Completion tokens: {self.completion_tokens}\n" \
               f"      Total tokens: {self.total_tokens}"
