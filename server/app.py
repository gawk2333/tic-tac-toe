import torch
from flask import Flask,request, jsonify
from torch import nn
import os


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

model = TicTacNet()


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        predictions = model.predict(data['input_data'])
        response = {
            'predictions': predictions
        }
        return jsonify(response)
    except Exception as e:
        error = str(e)
        return jsonify({'error': error}), 500

if __name__ == '__main__':
    app.run(debug=True)
