from flask import Flask
from flask import request

# Internal Modules
from modules.summarizer import Summarizer

smrz = Summarizer()

app = Flask(__name__)

@app.route("/summarize_text", methods=["POST"])
def summarize_text():
	req = request.get_json()
	res = smrz.run(req)
	return res

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5079, threaded=True, debug=True)