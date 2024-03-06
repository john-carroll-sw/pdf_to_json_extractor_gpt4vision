
class FileData:
    def __init__(self, processing_time, num_pages):
        self.processing_time = processing_time
        self.num_pages = num_pages

    def get_formatted_processing_time(self, file_name):
        minutes = int(self.processing_time // 60)
        seconds = int(self.processing_time % 60)
        return f"{file_name}  ({self.num_pages} pages): {minutes} minutes {seconds} seconds"
