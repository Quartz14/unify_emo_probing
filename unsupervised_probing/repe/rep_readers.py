from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
from itertools import islice
import torch

def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction  vector (d_2,)"""
    if type(direction) != torch.Tensor:
        H = torch.Tensor(H).cuda()
        direction = torch.Tensor(direction).to(H.device)
    mag = torch.norm(direction)
    assert not torch.isinf(mag).any()

    projection = H.matmul(direction) / mag  # normalized projection of H onto direction
    return projection

def recenter(x, mean=None):
    x = torch.Tensor(x).cuda()
    if mean is None:
        mean = torch.mean(x, axis=0, keepdims=True).cuda()
    else:
        mean = torch.Tensor(mean).cuda()
    return x-mean

class RepReader(ABC):
    @abstractmethod
    def __init__(self) -> None:
        self.direction_method = None
        self.directions = None 
        self.direction_signs = None 

    @abstractmethod
    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        pass

    def get_signs(self, hidden_states, train_choices, hidden_layers):
        """Given labels for training data hidden_states, determine whether the negative or positive directions corresponds to low/high concept"""
        signs = {}
        if self.needs_hiddens and hidden_states is not None and len(hidden_states)>0:
            for layer in hidden_layers:
                assert hidden_states[layer].shape[0] == 2 * len(train_choices), f"Shape mismatch bwt hidden states ({hidden_states[layer].shape[0]}) and labels ({len(train_choices)})"

                signs[layer] = []
                for component_index in range(self.n_components):
                    transformed_hidden_states = project_onto_direction(hidden_states[layer], self.directions[layer][component_index])
                    projected_scores = [transformed_hidden_states[i:i+2] for i in range(0, len(transformed_hidden_states), 2)]
                    outputs_min = [1 if min(o) == o[label] else 0 for o, label in zip(projected_scores, train_choices)]
                    outputs_max = [1 if max(o) == o[label] else 0 for o, label in zip(projected_scores, train_choices)]
                    signs[layer].append(-1 if np.mean(outputs_min)>np.mean(outputs_max) else 1)
        else:
            for layer in hidden_layers:
                signs[layer] = [1 for _ in range(self.n_components)]
        return signs
    
    def transform(self, hidden_states, hidden_layers, component_index):
        """Projects the hidden states onto the concept directions in self.directions"""
        assert component_index<self.n_components
        transformed_hidden_states = {}
        for layer in hidden_layers:
            layer_hidden_states = hidden_states[layer]
            
            if hasattr(self, 'H_train_means'):
                layer_hidden_states = recenter(layer_hidden_states, mean=self.H_train_means[layer])
            H_transformed = project_onto_direction(layer_hidden_states, self.directions[layer][component_index])
            transformed_hidden_states[layer] = H_transformed.cpu().numpy()
        return transformed_hidden_states
    
    def transform_crosslayer(self, hidden_states, hidden_layers, component_index=0):
        assert component_index<self.n_components
        transformed = {}

        for source_layer in hidden_states:
            vec = self.directions[source_layer][component_index]
            if hasattr(self, 'H_train_means'):
                mean_source = self.H_train_means[source_layer]
            else:
                mean_source = 0.0

            transformed[source_layer] = {}
            for target_layer in hidden_layers:
                layer_hidden_states = hidden_states[target_layer]
                layer_hidden_states = recenter(layer_hidden_states, mean=mean_source)
                H_transformed = project_onto_direction(layer_hidden_states, vec)
                transformed[source_layer][target_layer] = H_transformed.cpu().numpy()
        return transformed
    
class PCARepReader(RepReader):
    
    def __init__(self, n_components=1):
        super().__init__()
        self.n_components = n_components
        self.H_train_means = {}
        self.needs_hiddens = True
    
    def get_rep_directions(self, model, tokenizer, hidden_states, hidden_layers, **kwargs):
        directions = {}
        for layer in hidden_layers:
            H_train = hidden_states[layer]
            H_train_mean = H_train.mean(axis=0, keepdims=True)
            self.H_train_means[layer] = H_train_mean
            H_train = recenter(H_train, mean=H_train_mean).cpu()
            H_train = np.vstack(H_train)
            pca_model = PCA(n_components=self.n_components, whiten=False).fit(H_train)

            directions[layer] = pca_model.components_ #shape (n_components, n_features)
            self.n_components = pca_model.n_components_
        return directions
    
    def get_signs(self, hidden_states, train_labels, hidden_layers):
        signs = {}
        for layer in hidden_layers:
            assert hidden_states[layer].shape[0] == len(np.concatenate(train_labels)), f"Shape mismatch bwt hidden states ({hidden_states[layer].shape[0]}) and labels ({len(np.concatenate(train_labels))})" 
            layer_hidden_states = hidden_states[layer]
            
            layer_hidden_states = recenter(layer_hidden_states, mean=self.H_train_means[layer]) # Since scoring is comparative, effect of this is moot itseems
            
            layer_signs = np.zeros(self.n_components)
            for component_index in range(self.n_components):
                transformed_hidden_states = project_onto_direction(layer_hidden_states, self.directions[layer][component_index]).cpu()
                pca_outputs_comp = [list(islice(transformed_hidden_states, sum(len(c) for c in train_labels[:i]), sum(len(c) for c in train_labels[:i+1]))) for i in range(len(train_labels))]

                pca_outputs_min = np.mean([o[train_labels[i].index(1)] == min(o) for i,o in enumerate(pca_outputs_comp)])
                pca_outputs_max = np.mean([o[train_labels[i].index(1)] == max(o) for i,o in enumerate(pca_outputs_comp)])
                layer_signs[component_index] = np.sign(np.mean(pca_outputs_max) - np.mean(pca_outputs_min))
                if layer_signs[component_index] == 0:
                    layer_signs[component_index] = 1
            signs[layer] = layer_signs
        return signs


DIRECTION_FINDERS = {'pca': PCARepReader}

