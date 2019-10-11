from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/weight/sensitivity', methods=['POST'])
def weight_sensitivity():
	# content = request.get_json(silent=True)
	return jsonify({"success": True})