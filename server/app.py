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
model.load_state_dict(torch.load("target.pth"))

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()
        # Convert the input data (list) to a PyTorch tensor
        input_data = torch.tensor(data['board_status'], dtype=torch.float32)
        print(input_data)
        # Pass the tensor as input to the model
        predictions = torch.argmax(model(input_data))
        print(predictions.item())
        response = {
            'prediction': predictions.item()  # Convert the tensor to a list for JSON serialization
        }
        return jsonify(response)
    except Exception as e:
        error = str(e)
        return jsonify({'error': error}), 500
