from modules.q_and_a import Q_and_A
from modules.summarizer import Summarizer


request = {}
request["source"] = "snap"
request["file_name"] = "snap_10k_small.txt"
request["qestion"] = "What are the products being developed by SnapChat?"

qa = Q_and_A()
res = qa.run(request)


# smrz = Summarizer()
# res = smrz.run(request)

print(res)