from flask import Flask, request, render_template  
import pandas as pd

from rf import Rf

rf = Rf('sessions.jsonl')

app = Flask(__name__)

@app.route("/", methods =["GET", "POST"])
def gfg():
	if request.method == "POST":
		if "submit" in request.form:
			rawdata = request.form.get('rows')
			return render_template("form.html", predicted=get_prediction(rawdata))
		elif "compare" in request.form:
			data = compare_models()
			return render_template("form.html", precision1=data[0], precision2=data[1], diff=data[2])
	return render_template("form.html")

def compare_models():
	random_forest = Rf('sessions.jsonl')
	return random_forest.accuracy()

def get_prediction(rawdata):
	rawdata = rawdata.replace("'", '"')
	rawdata = rawdata.replace("None", '0')
	return rf.rf_predict(rawdata)


if __name__=='__main__':
	app.run() 
