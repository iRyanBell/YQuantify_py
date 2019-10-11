from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
  return 'Hello, World!'

@app.route('/server/health')
def server_health():
  return jsonify(success=True)