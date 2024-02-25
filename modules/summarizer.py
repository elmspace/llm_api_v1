import re
from transformers import pipeline
from transformers import AutoTokenizer

class Summarizer:


	def __init__(self):
		model_name = "Falconsai/text_summarization"
		self.model_token_seq_len = 512
		self.summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.summarizer_model = pipeline("summarization", model=model_name)


	def run(self, request):
		
		text = request["input"]
		clean_text = self.process_input(text)
		text_chunks = self.text_splitting(clean_text)
		print(text_chunks)
		# self.summarize_text(text)


	def process_input(self, input_text):
		"""
			Process and clean input-text for the summarization model.

			This method will process the input text to make sure it is
			ready for the model summarization.

			Parameters:
			- input_text (string): pre-processed text

			Returns:
			- clean_text (string) : processed text
		"""
		# Removing special characters
		clean_text = re.sub(r"[^\w\s]", "", input_text)
		# Remove any emoji characters
		clean_text = re.sub(r"[\U00010000-\U0010ffff]", "", clean_text)
		return clean_text
		

	def text_splitting(self, input_text):
		"""
			Split text into chunks.

			This method will split the text into chunks, so it is in the required
			token size length, required by the summarization model.

			Parameters:
			- input_text (string) : pre-chunked text

			Return:
			- text_chunks (list of strings) : list of text chunks
		"""
		text_chunks = []
		text_encoding = self.summarizer_tokenizer(input_text)
		if len(text_encoding["input_ids"]) > self.model_token_seq_len:
			pass
		else:
			text_decoded = self.summarizer_tokenizer.decode(text_encoding["input_ids"], skip_special_tokens=True)
			text_chunks.append(text_decoded)
		return text_chunks





	def summarize_text(self, text):


		summary = self.summarizer_model(text)

		print(summary)
