class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.W = nn.Linear(in_features, out_features)
        self.a = nn.Linear(2 * out_features, 1)

    def forward(self, X, A):
        h = self.W(X)
        N = h.size(0)
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * h.size(1))
        e = F.leaky_relu(self.a(a_input).squeeze(2), negative_slope=0.2)
        attention = F.softmax(e, dim=1)
        h_prime = torch.bmm(attention.unsqueeze(0), h.unsqueeze(0))
        return h_prime.squeeze(0)

  class GAT(nn.Module):
    def __init__(self, in_features, out_features, num_layers):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(in_features, out_features))
        for _ in range(num_layers - 1):
            self.layers.append(GATLayer(out_features, out_features))

    def forward(self, X, A):
        h = X
        for layer in self.layers:
            h = layer(h, A)
        return h

# Example usage
num_nodes = 100
input_dim = 64
output_dim = 16
num_layers = 2

# Create random input features and adjacency matrix
X = torch.randn(num_nodes, input_dim)
A = torch.randn(num_nodes, num_nodes)

import numpy as np
import tensorflow as tf
n = 50  # Number of nodes
p = 0.3  # Probability of edge existence

adjacency = np.random.choice([0, 1], size=(n, n), p=[1 - p, p])
adjacency = np.triu(adjacency) + np.triu(adjacency, 1).T  # Make adjacency matrix symmetric

# Generate random feature matrix
d = 10  # Dimensionality of features
features = np.random.rand(n, d)

# Convert to TensorFlow tensors
features = tf.convert_to_tensor(features, dtype=tf.float32)
adjacency = tf.convert_to_tensor(adjacency, dtype=tf.float32)

model = GAT(input_dim, output_dim, num_layers)
output = model(X, A)
print(output)

# Generate graph visualization
G = nx.from_numpy_array(adjacency.numpy())

# Get node positions for visualization
pos = nx.spring_layout(G)

# Draw nodes and edges
nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=200)
nx.draw_networkx_edges(G, pos)

# Add labels to nodes
labels = {i: str(i) for i in range(n)}
nx.draw_networkx_labels(G, pos, labels, font_color='black')

# Show the graph visualization
plt.title("Graph Visualization")
plt.axis('off')
plt.show()

