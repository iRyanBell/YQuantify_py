from flask import Flask, request, jsonify
import weight_sensitivity_analysis

app = Flask(__name__)


@app.route('/weight/sensitivity', methods=['POST'])
def weight_sensitivity():
    req_body = request.get_json(silent=True)
    key = req_body.key
    return jsonify(weight_sensitivity_analysis.perform_analysis(key=key))
