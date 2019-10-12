from flask import Flask, request, jsonify
# import weight_sensitivity

app = Flask(__name__)


@app.route('/weight/sensitivity', methods=['POST'])
def weight_sensitivity_analysis():
    key = request.form['key']
    return jsonify({'success': True, 'key': key})
