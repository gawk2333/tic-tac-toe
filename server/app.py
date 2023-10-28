import torch
from flask import Flask,request, jsonify
from torch import nn
# from flask_cors import CORS
from dotenv import load_dotenv
import os

load_dotenv()
class TicTacNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dl1 = nn.Linear(9, 36)
        self.dl2 = nn.Linear(36, 36)
        self.output_layer = nn.Linear(36, 9)
    def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)

        x = self.dl2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x

player1 = TicTacNet()
player1.load_state_dict(torch.load("target.pth"))
player2 = TicTacNet()
player2.load_state_dict(torch.load("target.pth"))

app = Flask(__name__,static_folder='./client/build/',static_url_path='/')
# CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        board = data['board_status']
        player = data['player']
        if player.sign == -1:
            board = [-x if x in (1, -1) else x for x in board]

        board = torch.tensor(board,dtype=torch.float32)

        if player == 'X':
            predictions = torch.argmax(player1(board))
        response = {
            'player': 'X',
            'action': predictions.item()  
        }
        
        if player == 'O':
            predictions = torch.argmax(player2(board))
        response = {
            'player': 'O',
            'action': predictions.item()
        }

        return jsonify(response)
    except Exception as e:
        error = str(e)
        return jsonify({'error': error}), 500

if __name__ == '__main__':
    app.run(port=(os.getenv('PORT') if os.getenv('PORT') else 8000), debug=False)
