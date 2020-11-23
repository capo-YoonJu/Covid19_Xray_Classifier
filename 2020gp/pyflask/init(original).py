import os, sys
from flask import Flask, escape, request,  Response, g, make_response
from flask.templating import render_template
from werkzeug.utils import secure_filename
from . import neural_style_transfer

app = Flask(__name__)
app.debug = True

# Main page
@app.route('/')
def index():
	return render_template('index.html')

@app.route('/nst_get')
def nst_get():
	return render_template('nst_get.html')

@app.route('/nst_post', methods=['GET','POST'])
def nst_post():
	if request.method == 'POST':
		# Reference Image
		refer_img = request.form['refer_img']
		refer_img_path = 'static/images/'+str(refer_img)

		# User Image (target image)
		user_img = request.files['user_img']
		user_img.save('./pyflask/static/images/'+str(user_img.filename))
		user_img_path = './static/images/'+str(user_img.filename)

		# Neural Style Transfer 
		transfer_img = neural_style_transfer.main(refer_img_path, user_img_path)
		transfer_img_path = './static/images/'+str(transfer_img.split('/')[-1])

	return render_template('nst_post.html', 
					refer_img=refer_img_path, user_img=user_img_path, transfer_img=transfer_img_path)