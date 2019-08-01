from flask import Flask, jsonify, make_response, request, abort, redirect
import logging
from processer import process
from processer import process_two_image
app = Flask(__name__)

@app.route('/')
def index():
    return redirect("http://tradersupport.club", code=302)

@app.route('/face_recognition', methods=['POST'])
def face_recognition():
    try:
        data = request.json
        result = process(data)
        return make_response(result, 200)
    except Exception as err:
        logging.error('An error has occurred whilst processing the file: "{0}"'.format(err))
        abort(400)

@app.route('/face_recognition_two_image', methods=['POST'])
def face_recognition_two_image():
    try:
        data = request.json
        result = process_two_image(data)
        return make_response(result, 200)
    except Exception as err:
        logging.error('An error has occurred whilst processing the file: "{0}"'.format(err))
        abort(400)

@app.errorhandler(400)
def bad_request(erro):
    return make_response(jsonify({'error': 'We cannot process the file sent in the request.'}), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Resource no found.'}), 404)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8084)