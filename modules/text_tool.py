import os
import re
import nltk
from modules.q_and_a import Q_and_A
from modules.summarizer import Summarizer
from nltk.tokenize import word_tokenize, sent_tokenize


class Text_Tool:


	def __init__(self):
		self.raw_data_path = "./raw_data/"

		# Helper objects
		self.summarizer = Summarizer()
		self.qa_agent = Q_and_A()


	def load_text(self, text_source, text_file_name="all"):
		"""
			Method for loading the text data.

			Thie method will load either a specific file, or collection of text files from a 
			source.

			Parameters:
			- text_source [string] : Source name for the text data
			- text_file_name [string] : File name or all.

			Returns:
			- text [string] : Text from the source file(s).
		"""
		if text_file_name == "all":
			raw_data_source = self.raw_data_path + text_source
			files_to_index = [raw_data_source + "/" + f for f in os.listdir(raw_data_source)]
			text_list = []
			for file in files_to_index:
				with open(file, "r") as text_file:
					text_list.append(text_file.read())
			text = " ".join(text_list)
		else:
			raw_text_file = self.raw_data_path + text_source + "/" + text_file_name
			with open(raw_text_file, "r") as text_file:
				text = text_file.read()
		return text


	def split_text_by(self, text, pattern):
		"""
			Method for splitting the text based on pattern

			Parameters:
			- text [string] : text data to be split.
			- pattern [string] : pattern to split the text by.

			Returns:
			- text_chunks [list] : list of text.
		"""
		text_chunks = text.split(pattern)
		text_chunks = [i.strip() for i in text_chunks]
		return text_chunks


	def summarize_text(self, text, summary_size="small"):
		"""
			Method creating summaries of text using Summarizer class

			Parameters:
			- text [string] : text data to be summarized.
			- summary_size [string] : size of the summary, small, medium or large. Default = small

			Returns:
			- text_chunks [list] : list of text.
		"""
		summaries = []
		if type(text) == list:
			for text_val in text:
				summaries.append(self.summarizer.run(text_val, summary_size))
		else:
			summaries.append(self.summarizer.run(text, summary_size))
		return summaries


	def answer_questions_w_context(self, question, context):
		"""
			Method uses the QA object and context to answer a question about the context.

			Parameters:
			- question [string] : Question to be answered using the context.
			- context [string] : Context text to be used to extract answer from.

			Returns:
			- response [string] : Response string.
		"""
		response = self.qa_agent.run_single(question, context)
		response = response["answer"]
		return response