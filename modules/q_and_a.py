import os
from transformers import pipeline
from haystack.nodes import FARMReader
from haystack.utils import print_answers
from haystack.nodes import BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline


class Q_and_A:


	def __init__(self):
		self.document_store = InMemoryDocumentStore(use_bm25=True)
		self.raw_data_path = "./raw_data/"
		self.data_store_path = "./data_store/"
		# Single reader mode
		model_name = "deepset/roberta-base-squad2"
		self.single_agent = pipeline('question-answering', model=model_name, tokenizer=model_name)


	def run_single(self, question, context):
		"""
			Main method for the signle context question and answering class.

			This method takes as input a <source> name and a <question>, and loads
			the relevant document and uses an LLM model to find the answe.

			Parameters:
			- question [string] : Question to be answered using the context.
			- context [string] : Context text to be used to extract answer from.

			Returns:
			- output [json] : Response object.
		"""
		input_data = {'question': question, 'context': context}
		output = self.single_agent(input_data)
		return output
		

	def run(self, request):
		"""
			Main method for the question and answering class.

			This method takes as input a <source> name and a <question>, and loads
			the relevant document and uses an LLM model to find the answe.

			Parameters:
			- request [json] : request json, containing information for the Q&A

			Returns:
			- result [json] : json object containing the answer of the input question.
		"""
		source = request["source"]
		
		self.index_files(source)

		retriever = self.get_retriever()
		reader = self.get_reader()

		pipe = ExtractiveQAPipeline(reader, retriever)

		prediction = pipe.run(
			query=request["qestion"],
			params={"Retriever": {"top_k": 30},"Reader": {"top_k": 1}}
		)
		result = {}
		answers = prediction["answers"]
		if len(answers) > 0:
			result["status"] = "pass"
			result["answer"] = answers[0].answer
		else:
			result["status"] = "fail"
			result["answer"] = "No answer was found"
		return result


	def get_reader(self, reader_name="roberta"):
		"""
			This method returns the reader model.
			
			Parameters:
			- reader_name [string] : Name of the reader model. By default it is "RoBERTA"

			Returns:
			- reader_model [FARM.node] : Framework for Adapting Representation Models (FARM).
		"""
		match reader_name:
			case "roberta":
				reader_model = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
			case _:
				reader_model = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=False)
		return reader_model
		

	def get_retriever(self, retriever_name="BM25"):
		"""
			This method returns the retriever model.
			
			Parameters:
			- retriever_name [string] : Name of the retriever model. By default it is "BM25"

			Returns:
			- retriever_model [haystack.node] : Retriever model.
		"""
		match retriever_name:
			case "BM25":
				retriever_model =  BM25Retriever(document_store=self.document_store)
			case _:
				retriever_model =  BM25Retriever(document_store=self.document_store)
		return retriever_model


	def index_files(self, source):
		"""
			This method performs indexing of the text file(s).

			Parameters:
			- source [string] : string representing the name of source data.

			Returns:
			- N/A [None] : No result is returned.
		"""
		raw_data_source = self.raw_data_path + source
		data_store_source = self.data_store_path + source

		files_to_index = [raw_data_source + "/" + f for f in os.listdir(raw_data_source)]
		
		indexing_pipeline = TextIndexingPipeline(self.document_store)
		indexing_pipeline.run_batch(file_paths=files_to_index)