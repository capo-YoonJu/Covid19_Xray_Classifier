import os, sys
from flask import Flask, escape, request,  Response, g, make_response
from flask.templating import render_template
from werkzeug.utils import secure_filename
from . import neural_style_transfer
from . import g_prediction

app = Flask(__name__, static_url_path='/static')
app.debug = True

# Main page
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/img_select')
def img_select():
	return render_template('img_select.html')

@app.route('/result', methods=['GET','POST'])
def result():
	if request.method == 'POST':
		# User Name
		user_name = request.form['pname']
		
		# User Name
		user_birth = request.form['pbirth']
		
		# User Name
		user_tel = request.form['ptel']
	
	
		# User Image (target image)
		user_img = request.files['user_img']
		user_img.save('./pyflask/static/pyimages/'+str(user_img.filename))
		user_img_path = './static/pyimages/'+str(user_img.filename)
		user_img_path2 = './pyimages/'+str(user_img.filename)
		
		# GoogLeNet Prediction 
		pred_class = g_prediction.main(user_img_path)
		
		if pred_class == 0:
			str_class = "정상"
		elif pred_class == 1:
			str_class = "코로나"
		elif pred_class == 2:
			str_class = "박테리아"
		elif pred_class == 3:
			str_class = "바이러스"
			
	return render_template('result.html', 
					p_name=user_name, p_birth=user_birth, p_tel=user_tel, refer_img=user_img_path2, user_img=user_img_path2, transfer_img=user_img_path2, pred=str_class)
					
@app.route('/patient_progress')
def patient_progress():
	return render_template('patient_progress.html')