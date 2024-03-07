import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from modules.text_tool import Text_Tool
from sentence_transformers import SentenceTransformer


txttool = Text_Tool()

text = txttool.load_text(text_source="snap", text_file_name="snap-10k.txt")
text_chunks = txttool.split_text_by(text, pattern=".")

# res = txttool.summarize_text(text_chunks[0], summary_size="large")


# question = "Which companies are the closest competitor?"

# - Use LLM to get back text on what Snap's comps are
# - Use the text returned to NIR - name of the company
# - Search in the document for exact matches of the name

# res = txttool.answer_questions_w_context(question, text_chunks[0])


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

base_txt = "Company abcd is competitor to Snap."
base_emb = model.encode(base_txt)

embs = []
for stc in tqdm(text_chunks[0:10]):
	embs.append(model.encode(stc))


embs = np.array(embs)

res = cosine_similarity([base_emb], embs)

