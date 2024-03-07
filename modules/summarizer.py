import re
import os
from transformers import pipeline
from transformers import AutoTokenizer

class Summarizer:


	def __init__(self):
		"""
			Class constructor.
		"""
		model_name = "facebook/bart-large-cnn"
		self.raw_data_path = "./raw_data/"
		self.model_token_seq_len = 500
		self.summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.summarizer_model = pipeline("summarization", model=model_name)


	def run(self, text, summary_size):
		"""
			Main method for the summarizer class.

			This method takes as input a request json, with the text to be summarized
			in the input key. It then returns the summary of the text.

			Parameters:
			- text [string] : text to be summarize
			- summary_size[string] : default = small, size of the summary text

			Returns:
			- text [string] : summary text.
		"""
		try:
			while True:
				text_chunks = self.text_splitting(text)
				summary = self.summarize_text(text_chunks)
				if (summary_size=="small") and (len(summary) == 1):
					summary = summary[0]
					break
				elif (summary_size=="medium") and (len(summary) <= 5):
					summary = " ".join(summary)
					break
				elif (summary_size=="large") and (len(summary) <= 10):
					summary = " ".join(summary)
					break
				else:
					text = " ".join(summary).strip()
		except Exception as e:
			print(str(e))
		return summary


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
		# Remove any emoji characters
		clean_text = re.sub(r"[\U00010000-\U0010ffff]", "", input_text)
		return clean_text
		

	def text_splitting(self, input_text):
		"""
			Split text into chunks.

			This method will split the text into chunks, so it is in the required
			token size length, required by the summarization model.

			Parameters:
			- input_text (string) : pre-chunked text

			Return:
			- text_chunks (list) : list of text chunks
		"""
		text_chunks = []
		pattern = r'(?<=[.?!])'
		splitted_text = re.split(pattern, input_text)
		splitted_text = [s.strip() for s in splitted_text if s]
		current_text_blob = []
		for text_chunk in splitted_text:
			current_text_blob.append(text_chunk)
			blob_seq_size = self.compute_blob_seqence_size(current_text_blob)
			if blob_seq_size > self.model_token_seq_len:
				text = " ".join(current_text_blob[0:len(current_text_blob)-1]).strip()
				text = self.process_input(text)
				text_chunks.append(text)
				while blob_seq_size > self.model_token_seq_len:
					# current_text_blob.pop(0)
					current_text_blob = []
					blob_seq_size = self.compute_blob_seqence_size(current_text_blob)
		if len(text_chunks) == 0:
			text = " ".join(current_text_blob).strip()
			text = self.process_input(text)
			text_chunks.append(text)
		return text_chunks


	def compute_blob_seqence_size(self, text_list):
		"""
			Compute the sequence length of text blob

			This method will compute how many tokens (based on the type of the model)
			the given text blob is.

			Parameters:
			- input_list (list) : list of strings

			Return:
			- blob_size (int) : the number tokens in the sequence
		"""
		text = " ".join(text_list)
		encoded_text = self.summarizer_tokenizer(text)
		blob_size = len(encoded_text["input_ids"])
		return blob_size


	def summarize_text(self, text_chunks):
		"""
			Summarise text

			This method uses the pre-trained LLM model to summarize a list of texts.
			The summarizes are then returned as a list of text.

			Parameters:
			- text_chunks (list) : list of texts to be summarized.

			Return:
			- summary (list) : list of summaries of the input text.
		"""
		summaries = self.summarizer_model(text_chunks)
		summary = [x["summary_text"] for x in summaries]
		return summary


