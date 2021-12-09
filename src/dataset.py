import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, numerical_features, prediction_positions):
        self.encodings = encodings
        self.labels = labels
        self.numerical_features = numerical_features
        self.prediction_positions = prediction_positions

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).to(torch.float32)
        # add numerical features: utterance feature vectors (specified metrics per utterance) concatenated sequentially
        # + prediction position as scalar
        item['numerical_features'] = torch.tensor(self.numerical_features[idx])  # get metrics
        item['numerical_features'] = item['numerical_features'].to(torch.float32)  # convert to float32
        item['prediction_position'] = torch.tensor([self.prediction_positions[idx]])
        item['ids'] = torch.tensor([idx])
        return item

    def __len__(self):
        return len(self.labels)
