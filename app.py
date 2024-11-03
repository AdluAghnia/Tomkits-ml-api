from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    data = {
        "message" : "Hello World",
    }
    return jsonify(data)

if __name__ == "__main__":
    app.run(debug=True)
