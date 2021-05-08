from flask import Flask, url_for, send_from_directory, request, redirect
import logging, os
from werkzeug.utils import secure_filename
import recognize_doc
import cv2
from pancard.pancard_main import pan_validate
app = Flask(__name__)
file_handler = logging.FileHandler('server.log')
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

PROJECT_HOME = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = '{}/uploads/'.format(PROJECT_HOME)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def create_new_folder(local_dir):
	newpath = local_dir
	if not os.path.exists(newpath):
		os.makedirs(newpath)
	return newpath

def aadhaar_validation(img_path,name):
	validity=recognize_doc.recognize(img_path,name)
	print(validity)
	return validity

def pan_card_validation(img_path):
	validity= pan_validate(img_path)
	print(validity)
	return validity

@app.route('/validation', methods = ['POST'])
def api_root():
	app.logger.info(PROJECT_HOME)
	if request.method == 'POST' and request.files['image']:
		app.logger.info(app.config['UPLOAD_FOLDER'])
		img = request.files['image']
		name = request.form.get('name')
		print(name)
		img_name = secure_filename(img.filename)
		create_new_folder(app.config['UPLOAD_FOLDER'])
		saved_path = os.path.join(app.config['UPLOAD_FOLDER'], img_name)
		app.logger.info("saving {}".format(saved_path))
		img.save(saved_path)
		doc_type = request.form.get('document')
		if doc_type=='aadhaar':
			validity=aadhaar_validation(saved_path,name)
			return validity
		if doc_type=='pancard':
			validity=pan_card_validation(saved_path)
			return validity
		if doc_type=='voterid':
			return "| Waiting for Code |"
		else:
			return "Invalid Document type"
	else:
		return "Where is the image?"

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True)
