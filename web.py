from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "Bot activo OK", 200
