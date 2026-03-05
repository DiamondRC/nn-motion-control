import torch
from json_manager import load_config
from mlp import MLP

MODEL_SAVE_PATH = (
    "src/deltabot_nn_controller/model_zoo/models/model_states/best_model.pth"
)


class HeatmapModel:
    """
    Utility class for visualizing model performance with heatmaps.

    Opens a saved model and displays the matrix of weights and
    biases as a heatmap.

    Useful for diagnosing issues like vanishing/exploding gradients,
    dead neurons, as well as for compression analysis for FPGA deployment.
    """

    def __init__(self, model_config, save_path):
        self.model = MLP(model_config)
        self.save_path = save_path

    def _plot_heatmap(self, weights, biases):
        pass
        # Plot heatmaps for weights and biases
        # for idx, weight in enumerate(weights):

        # plt.figure(figsize=(8, 6))
        # plt.imshow(data, cmap="viridis", aspect="auto")
        # plt.colorbar()
        # plt.title(title)
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.show()

    def visualize_model(self):
        # Load model state onto CPU for analysis
        state_dict = torch.load(self.save_path, map_location="cpu")
        self.model.load_state_dict(state_dict)

        # Extract weights and biases
        weights = []
        biases = []
        for name, param in self.model.named_parameters():
            if "weight" in name:
                weights.append(param.detach().numpy())
            elif "bias" in name:
                biases.append(param.detach().numpy())

        # Pass to plotting function
        self._plot_heatmap(weights, biases)


# Need to extract number of layers in model
# Need to extract the length of each model row

HeatmapModel(load_config(), MODEL_SAVE_PATH).visualize_model()
