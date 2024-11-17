# HIGH-LEVEL PSEUDOCODE/BASELINE CODE FOR KANG (CPU)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter_add
from scipy.interpolate import BSpline
import seaborn as sns
import matplotlib.pyplot as plt
import random

class KANEdgeFunction(nn.Module):
    def __init__(self, hidden_dim, grid_size=10, spline_degree=3, relation_property=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.spline_degree = spline_degree
        self.relation_property = relation_property
        
        # Base function parameters
        self.w_base = nn.Parameter(torch.randn(1))
        self.base_activation = nn.Sigmoid()
        
        # Spline function parameters
        self.w_spline = nn.Parameter(torch.randn(1))
        self.spline_coeffs = nn.Parameter(torch.randn(grid_size))
        
        # Knots for B-spline basis
        self.knots = torch.linspace(-5, 5, grid_size)
        
        # History for visualization
        self.function_history = []
        
    def compute_bspline_basis(self, x):
        """Compute B-spline basis functions for given input."""
        # Handle both single values and batched inputs
        if x.dim() == 0:
            x = x.unsqueeze(0)
        
        # Create grid of basis functions
        basis = torch.zeros(x.shape[0], self.grid_size, device=x.device)
        
        # Compute normalized distances to knots
        for i in range(self.grid_size):
            # Compute distance to current knot
            t = (x - self.knots[i]) / (self.knots[1] - self.knots[0])
            
            # B-spline basis (piece-wise polynomial)
            # Use smooth approximation for better gradients
            t_clamped = torch.clamp(t, 0, 1)
            basis[:, i] = (1 - t_clamped)**self.spline_degree * (t_clamped < 1).float()
        
        # Normalize basis functions
        basis = F.softmax(basis, dim=1)
        return basis

    def forward(self, x):
        """Forward pass through the edge function.
        Args:
            x: Node features [hidden_dim]
        Returns:
            out: Transformed features [hidden_dim]
        """
        # Ensure input is properly shaped
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Base function component
        base_out = self.w_base * self.base_activation(x)  # [batch_size, hidden_dim]
        
        # Spline component
        x_mean = x.mean(dim=-1)  # [batch_size]
        basis = self.compute_bspline_basis(x_mean)  # [batch_size, grid_size]
        spline_out = self.w_spline * torch.matmul(basis, self.spline_coeffs.unsqueeze(-1))  # [batch_size, 1]
        spline_out = spline_out.expand_as(base_out)  # [batch_size, hidden_dim]
        
        # Combined output with smooth transition
        gate = torch.sigmoid(x_mean).unsqueeze(-1)  # [batch_size, 1]
        out = gate * base_out + (1 - gate) * spline_out  # [batch_size, hidden_dim]
        
        # Store for visualization/analysis
        with torch.no_grad():
            self.function_history.append({
                'input': x.detach(),
                'output': out.detach(),
                'base': base_out.detach(),
                'spline': spline_out.detach(),
                'gate': gate.detach()
            })
        
        # Return the first element if single input
        if out.size(0) == 1:
            out = out.squeeze(0)
        
        return out
    
    def get_complexity(self):
        """Compute function complexity using entropy of spline coefficients."""
        normalized_coeffs = F.softmax(self.spline_coeffs, dim=0)
        entropy = -(normalized_coeffs * torch.log(normalized_coeffs + 1e-10)).sum()
        return -entropy  # Negative entropy as complexity measure

class KANG(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_relations=5, num_layers=2, grid_size=10, spline_degree=3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.num_layers = num_layers
        
        # Node embedding layers
        self.node_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # Edge functions for each relation type
        self.edge_functions = nn.ModuleList([
            KANEdgeFunction(hidden_dim, grid_size, spline_degree, 'symmetric' if i % 2 == 0 else 'transitive')
            for i in range(num_relations)
        ])
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=0)
        )
        
        # Layer normalization for each layer
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.evolution_history = {}
        self.l1_lambda = 0.01
        
    def compute_attention(self, query, key):
        """Compute attention scores between query and key nodes.
        Args:
            query: [batch_size, hidden_dim]
            key: [batch_size, hidden_dim]
        Returns:
            attention: [batch_size]
        """
        # Ensure inputs are 2D
        if query.dim() == 3:
            query = query.squeeze(1)
        if key.dim() == 3:
            key = key.squeeze(1)
            
        # Compute attention scores
        combined = torch.cat([query, key], dim=-1)  # [batch_size, 2*hidden_dim]
        return self.attention(combined).squeeze(-1)  # [batch_size]
        
    def message_passing(self, h, edge_index, edge_type):
        """Implement message passing with attention and relation-specific transformations."""
        src, dst = edge_index
        num_nodes = h.size(0)
        hidden_dim = h.size(1)
        messages = torch.zeros(num_nodes, hidden_dim, device=h.device)
        attention_weights = []
        message_info = []
        
        # Group edges by destination node
        for node in range(num_nodes):
            # Find incoming edges
            mask = dst == node
            if not mask.any():
                continue
                
            # Get source nodes and their types
            sources = src[mask]
            types = edge_type[mask]
            num_neighbors = len(sources)
            
            # Get source node embeddings
            source_h = h[sources]  # [num_neighbors, hidden_dim]
            
            # Apply relation-specific transformations
            transformed_messages = []
            for i, (source, type_idx) in enumerate(zip(sources, types)):
                # Apply edge function
                edge_fn = self.edge_functions[type_idx]
                message = edge_fn(source)  # [hidden_dim]
                transformed_messages.append(message)
            
            if transformed_messages:  # Only process if there are messages
                transformed_messages = torch.stack(transformed_messages)  # [num_neighbors, hidden_dim]
                
                # Prepare query for attention
                query = h[node].unsqueeze(0).expand(num_neighbors, -1)  # [num_neighbors, hidden_dim]
                
                # Compute attention weights
                attn = self.compute_attention(query, transformed_messages)  # [num_neighbors]
                attention_weights.append(attn)
                
                # Apply attention to messages
                weighted_messages = attn.unsqueeze(-1) * transformed_messages  # [num_neighbors, hidden_dim]
                node_message = weighted_messages.sum(dim=0)  # [hidden_dim]
                messages[node] = node_message
                
                # Store message passing information
                message_info.append({
                    'node': node,
                    'sources': sources.tolist(),
                    'attention': attn.detach().tolist(),
                    'relation_types': types.tolist()
                })
        
        self.evolution_history['message_info'] = message_info
        return messages, attention_weights
    
    def forward(self, x, edge_index, edge_type):
        """Forward pass through the KANG model."""
        # Initial node embeddings
        h = self.node_embedding(x)  # [num_nodes, hidden_dim]
        
        # Multi-layer message passing
        attention_weights = []
        for layer in range(self.num_layers):
            # Message passing with attention
            messages, layer_attention = self.message_passing(h, edge_index, edge_type)
            attention_weights.extend(layer_attention)
            
            # Update node embeddings
            h = h + messages  # Residual connection
            h = self.layer_norms[layer](h)  # Layer normalization
            h = F.relu(h)  # Non-linearity
        
        return h, attention_weights
    
    def score_edges(self, node_embeddings, edge_index):
        """Score edges using relation-aware similarity."""
        src, dst = edge_index
        src_embeds = node_embeddings[src]
        dst_embeds = node_embeddings[dst]
        
        # Compute similarity scores
        scores = F.cosine_similarity(src_embeds, dst_embeds, dim=1)
        return torch.sigmoid(scores)
    
    def compute_loss(self, pos_score, neg_score, entropy_lambda=0.01):
        # Task loss (BCE)
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
        task_loss = F.binary_cross_entropy(scores, labels)
        
        # L1 regularization of edge functions
        l1_loss = sum(torch.abs(fn.w_base).mean() + torch.abs(fn.w_spline).mean() + 
                     torch.abs(fn.spline_coeffs).mean() for fn in self.edge_functions)
        
        # Entropy regularization
        entropy_loss = sum(fn.get_complexity() for fn in self.edge_functions)
        
        return task_loss + self.l1_lambda * l1_loss + entropy_lambda * entropy_loss
    
    def compute_metrics(self, pos_score, neg_score):
        """Compute evaluation metrics."""
        metrics = {
            'pos_score_mean': pos_score.mean().item(),
            'neg_score_mean': neg_score.mean().item(),
            'pos_score_std': pos_score.std().item(),
            'neg_score_std': neg_score.std().item()
        }
        
        # Compute relation importance
        importance = {}
        for i, fn in enumerate(self.edge_functions):
            l1_norm = torch.abs(fn.w_base).mean() + torch.abs(fn.w_spline).mean() + torch.abs(fn.spline_coeffs).mean()
            importance[f'relation_{i}_importance'] = l1_norm.item()
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        for k in importance:
            importance[k] /= total_importance
            metrics[k] = importance[k]
        
        return metrics

def train_kang(kang, x, edge_index, edge_type, pos_edge_index, neg_edge_index, lr=0.001, l1_lambda=0.01, entropy_lambda=0.01, property_lambda=0.1):
    print("\nStarting KANG training...")
    print(f"Network parameters: {sum(p.numel() for p in kang.parameters())} total parameters")
    print(f"Input features: {x.shape[1]}, Number of nodes: {x.shape[0]}")
    print(f"Number of edges: {edge_index.shape[1]}")
    
    optimizer = torch.optim.Adam(kang.parameters(), lr=lr)
    device = next(kang.parameters()).device
    
    # Lists to track metrics
    history = {
        'total_loss': [],
        'primary_loss': [],
        'reg_loss': [],
        'property_loss': [],
        'pos_score_mean': [],
        'neg_score_mean': [],
        'pos_score_std': [],
        'neg_score_std': [],
        'epoch': []
    }
    
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        node_embeddings, attention_weights = kang(x, edge_index, edge_type)
        
        # Calculate scores
        pos_score = kang.score_edges(node_embeddings, pos_edge_index)
        neg_score = kang.score_edges(node_embeddings, neg_edge_index)
        
        # Calculate losses
        primary_loss = kang.compute_loss(pos_score, neg_score, entropy_lambda=entropy_lambda)
        reg_loss = sum(torch.abs(fn.w_base).mean() + torch.abs(fn.w_spline).mean() + 
                     torch.abs(fn.spline_coeffs).mean() for fn in kang.edge_functions)
        property_loss = 0.0
        
        for i, edge_fn in enumerate(kang.edge_functions):
            if edge_fn.relation_property == 'symmetric':
                # For symmetric relations, penalize asymmetric edge functions
                x_vals = torch.linspace(-5, 5, 100, device=device)
                y = edge_fn(x_vals)
                flipped_y = torch.flip(y, dims=[0])
                property_loss += F.mse_loss(y, flipped_y)
            
            elif edge_fn.relation_property == 'transitive':
                # For transitive relations, encourage monotonicity
                x_vals = torch.linspace(-5, 5, 100, device=device)
                y = edge_fn(x_vals)
                # Penalize negative slopes
                diff = y[1:] - y[:-1]
                property_loss += torch.relu(-diff).mean()
        
        # Total loss
        total_loss = primary_loss + l1_lambda * reg_loss + property_lambda * property_loss
        
        # Backward pass and optimization
        total_loss.backward()
        optimizer.step()
        
        # Record metrics
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Total Loss: {total_loss.item():.4f}")
            print(f"Primary Loss: {primary_loss.item():.4f}")
            print(f"Regularization Loss: {reg_loss.item():.4f}")
            print(f"Property Loss: {property_loss.item():.4f}")
            print(f"Positive Score Mean: {pos_score.mean().item():.4f}")
            print(f"Negative Score Mean: {neg_score.mean().item():.4f}")
            
            history['total_loss'].append(total_loss.item())
            history['primary_loss'].append(primary_loss.item())
            history['reg_loss'].append(reg_loss.item())
            history['property_loss'].append(property_loss.item())
            history['pos_score_mean'].append(pos_score.mean().item())
            history['neg_score_mean'].append(neg_score.mean().item())
            history['pos_score_std'].append(pos_score.std().item())
            history['neg_score_std'].append(neg_score.std().item())
            history['epoch'].append(epoch)
    
    print("\nTraining completed!")
    
    # Create visualization data
    metrics = pd.DataFrame(history)
    
    # Collect attention weights
    attn = pd.DataFrame({
        'weight': torch.cat(attention_weights).cpu().numpy(),
        'layer': [i // len(attention_weights) for i in range(len(torch.cat(attention_weights)))]
    })
    
    # Collect function evolution history
    history_data = []
    for i, edge_fn in enumerate(kang.edge_functions):
        for t, h in enumerate(edge_fn.function_history):
            history_data.append({
                'time': t,
                'relation': i,
                'input': h['input'].cpu().numpy(),
                'output': h['output'].cpu().numpy(),
                'base': h['base'].cpu().numpy(),
                'spline': h['spline'].cpu().numpy(),
                'gate': h['gate'].cpu().numpy(),
            })
    
    history_df = pd.DataFrame(history_data)
    
    return metrics, attn, history_df

# Example usage
print("Loading citation network dataset...")
dataset = torch.load('citation_network_dataset.pt')

print("\nInitializing KANG model...")
kang = KANG(
    input_dim=dataset['node_features'].size(1),
    hidden_dim=64,
    num_relations=4,  # background, methodology, results, discussion
    num_layers=2,
    grid_size=10,
    spline_degree=3
)

# Split edges into train and test
train_edges = dataset['edge_index'][:, dataset['train_mask']]
train_types = dataset['edge_type'][dataset['train_mask']]
test_edges = dataset['edge_index'][:, ~dataset['train_mask']]
test_types = dataset['edge_type'][~dataset['train_mask']]

# Create negative edges for training
def create_negative_edges(edge_index, num_nodes, num_samples):
    pos_edges = set(map(tuple, edge_index.t().tolist()))
    neg_edges = []
    while len(neg_edges) < num_samples:
        i = random.randint(0, num_nodes-1)
        j = random.randint(0, num_nodes-1)
        if i != j and (i, j) not in pos_edges:
            neg_edges.append([i, j])
    return torch.tensor(neg_edges).t()

num_nodes = dataset['node_features'].size(0)
train_neg_edges = create_negative_edges(train_edges, num_nodes, train_edges.size(1))
test_neg_edges = create_negative_edges(test_edges, num_nodes, test_edges.size(1))

print("\nStarting training process...")
metrics, attn, history_df = train_kang(
    kang,
    dataset['node_features'],
    train_edges,
    train_types,
    train_edges,
    train_neg_edges,
    lr=0.001,
    l1_lambda=0.01,
    property_lambda=0.1
)

# Evaluate on test set
print("\nEvaluating on test set...")
with torch.no_grad():
    node_embeddings, _ = kang(dataset['node_features'], test_edges, test_types)
    pos_score = kang.score_edges(node_embeddings, test_edges)
    neg_score = kang.score_edges(node_embeddings, test_neg_edges)
    
    test_metrics = kang.compute_metrics(pos_score, neg_score)
    print("\nTest Set Metrics:")
    print(f"Positive Score Mean: {test_metrics['pos_score_mean']:.4f}")
    print(f"Negative Score Mean: {test_metrics['neg_score_mean']:.4f}")
    print(f"Score Difference: {(test_metrics['pos_score_mean'] - test_metrics['neg_score_mean']):.4f}")

# Additional visualization for citation patterns
plt.figure(figsize=(10, 6))
edge_type_counts = torch.bincount(dataset['edge_type'])
plt.bar(['Background', 'Methodology', 'Results', 'Discussion'], 
        edge_type_counts.numpy())
plt.title('Distribution of Citation Types')
plt.ylabel('Number of Citations')
plt.savefig('citation_type_distribution.png')
plt.close()

print("\nVisualization of citation type distribution saved as 'citation_type_distribution.png'")