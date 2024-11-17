import numpy as np
import torch
import networkx as nx
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer
import random

class CitationNetworkGenerator:
    def __init__(self, num_papers=1000, num_fields=10, embedding_dim=32, seed=42):
        self.num_papers = num_papers
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.seed = seed
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Edge types
        self.EDGE_TYPES = {
            'BACKGROUND': 0,    # Cites for background information
            'METHODOLOGY': 1,   # Cites for methodology
            'RESULTS': 2,       # Cites for results comparison
            'DISCUSSION': 3     # Cites for discussion/implications
        }
        
        # Initialize sentence transformer for generating paper embeddings
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def generate_knowledge_graph(self):
        """Generate a knowledge graph of research fields and their relationships."""
        kg = nx.DiGraph()
        
        # Create field nodes with properties
        field_names = [f"Field_{i}" for i in range(self.num_fields)]
        field_descriptions = [
            f"Research area focusing on {name.lower()} with various applications"
            for name in field_names
        ]
        
        # Add nodes to knowledge graph
        for i, (name, desc) in enumerate(zip(field_names, field_descriptions)):
            # Generate field embedding from its description
            embedding = self.encoder.encode(desc)
            kg.add_node(i, name=name, description=desc, embedding=embedding)
        
        # Add edges between related fields (hierarchical and peer relationships)
        for i in range(self.num_fields):
            # Each field is related to 2-3 other fields
            num_relations = random.randint(2, 3)
            related_fields = random.sample([j for j in range(self.num_fields) if j != i], num_relations)
            
            for j in related_fields:
                relation_type = random.choice(['hierarchical', 'peer'])
                kg.add_edge(i, j, type=relation_type)
        
        return kg, field_names
    
    def generate_paper_features(self, kg, field_names):
        """Generate paper features including titles, abstracts, and embeddings."""
        papers = []
        
        for i in range(self.num_papers):
            # Assign primary field
            primary_field = random.choice(range(self.num_fields))
            
            # Generate synthetic title and abstract
            title = f"Paper on {field_names[primary_field]} - Study {i}"
            abstract = f"This research investigates {field_names[primary_field].lower()} "
            abstract += f"with implications for {random.choice(field_names).lower()}. "
            abstract += "The study presents novel findings and methodological advances."
            
            # Generate embedding using the title and abstract
            text_embedding = self.encoder.encode(title + " " + abstract)
            
            # Add some field-specific patterns to the embedding
            field_embedding = kg.nodes[primary_field]['embedding']
            combined_embedding = 0.7 * text_embedding + 0.3 * field_embedding
            
            # Normalize the embedding
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)
            
            papers.append({
                'id': i,
                'title': title,
                'abstract': abstract,
                'primary_field': primary_field,
                'embedding': combined_embedding,
                'year': 2020 + random.randint(0, 3)  # Papers from 2020-2023
            })
        
        return papers
    
    def generate_citation_network(self, papers):
        """Generate citation network with different types of citations."""
        # Create graph
        G = nx.DiGraph()
        
        # Add paper nodes
        for paper in papers:
            G.add_node(paper['id'], 
                      title=paper['title'],
                      primary_field=paper['primary_field'],
                      year=paper['year'])
        
        # Generate citations
        edge_index = []
        edge_type = []
        
        for paper in papers:
            # Papers can only cite older papers
            potential_citations = [p for p in papers 
                                if p['year'] < paper['year'] and p['id'] != paper['id']]
            
            if not potential_citations:
                continue
            
            # Number of citations based on similarity and randomness
            num_citations = random.randint(3, 8)
            cited_papers = random.sample(potential_citations, 
                                      min(num_citations, len(potential_citations)))
            
            for cited_paper in cited_papers:
                # Determine citation type based on papers' relationship
                if paper['primary_field'] == cited_paper['primary_field']:
                    # Same field citations are more likely to be methodology or results
                    citation_type = random.choice([
                        self.EDGE_TYPES['METHODOLOGY'],
                        self.EDGE_TYPES['RESULTS'],
                        self.EDGE_TYPES['METHODOLOGY'],
                        self.EDGE_TYPES['RESULTS'],
                        self.EDGE_TYPES['DISCUSSION']
                    ])
                else:
                    # Different field citations are more likely to be background
                    citation_type = random.choice([
                        self.EDGE_TYPES['BACKGROUND'],
                        self.EDGE_TYPES['BACKGROUND'],
                        self.EDGE_TYPES['DISCUSSION'],
                        self.EDGE_TYPES['METHODOLOGY']
                    ])
                
                edge_index.append([paper['id'], cited_paper['id']])
                edge_type.append(citation_type)
                
                # Add edge to networkx graph
                G.add_edge(paper['id'], cited_paper['id'], type=citation_type)
        
        return G, torch.tensor(edge_index).t(), torch.tensor(edge_type)
    
    def generate_dataset(self):
        """Generate complete citation network dataset."""
        print("Generating knowledge graph...")
        kg, field_names = self.generate_knowledge_graph()
        
        print("Generating paper features...")
        papers = self.generate_paper_features(kg, field_names)
        
        print("Generating citation network...")
        citation_graph, edge_index, edge_type = self.generate_citation_network(papers)
        
        # Convert paper features to tensor
        node_features = torch.tensor(np.array([p['embedding'] for p in papers]), 
                                   dtype=torch.float32)
        
        # Create train/test split for edge prediction
        num_edges = edge_index.size(1)
        train_mask = torch.zeros(num_edges, dtype=torch.bool)
        train_mask[:int(0.8 * num_edges)] = True
        
        # Shuffle the mask
        perm = torch.randperm(num_edges)
        train_mask = train_mask[perm]
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'train_mask': train_mask,
            'papers': papers,
            'knowledge_graph': kg,
            'citation_graph': citation_graph,
            'field_names': field_names
        }

if __name__ == "__main__":
    # Generate dataset
    generator = CitationNetworkGenerator(num_papers=1000, num_fields=10)
    dataset = generator.generate_dataset()
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print(f"Number of papers: {len(dataset['papers'])}")
    print(f"Number of citations: {dataset['edge_index'].size(1)}")
    print(f"Number of fields: {len(dataset['field_names'])}")
    print(f"Node feature dimension: {dataset['node_features'].size(1)}")
    
    # Save dataset
    torch.save(dataset, 'citation_network_dataset.pt')
    print("\nDataset saved to 'citation_network_dataset.pt'")
