from flask import Flask, request, jsonify
import weight_sensitivity

app = Flask(__name__)


@app.route('/weight/sensitivity', methods=['POST'])
def weight_sensitivity_analysis():
    form_data = request.get_json()
    key = form_data['key']
    results = weight_sensitivity.perform_analysis(key=key)
    return jsonify({'results': results})
