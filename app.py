import json
import pandas as pd
from flask import Flask
from flask import request
from flask import Response
from cachetools import TTLCache
from flask import make_response

# Internal Modules
from modules.summarizer import Summarizer

smrz = Summarizer()

app = Flask(__name__)
cache = TTLCache(maxsize=500, ttl=60)

@app.route("/summarize_text", methods=["POST"])
def summarize_text():
	req = request.get_json()
	res = smrz.run(req)
	return res


if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5079, threaded=True, debug=True)