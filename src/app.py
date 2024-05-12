from http import HTTPStatus

from flask import Flask, request, jsonify
from flask_cors import CORS

from service import move_service

app = Flask(__name__)
CORS(app)


@app.route('/api/move', methods=['POST'])
def get_computer_move():
    data = request.get_json()
    response = move_service.get_computer_move(data)
    return jsonify(response)


@app.route('/api/rate_move', methods=['POST'])
def save_move_rating():
    data = request.get_json()
    move_service.save_move_rating(data)
    return "Move rated successfully", HTTPStatus.OK


@app.route('/api/eval', methods=['POST'])
def eval_board():
    data = request.get_json()
    response = move_service.get_stockfish_evaluation(data)
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
