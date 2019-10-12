from flask import Flask, request, jsonify
import weight_sensitivity_analysis

app = Flask(__name__)


@app.route('/weight/sensitivity', methods=['POST'])
def weight_sensitivity():
    key = request.form.key
    return jsonify(weight_sensitivity_analysis.perform_analysis(key=key))
