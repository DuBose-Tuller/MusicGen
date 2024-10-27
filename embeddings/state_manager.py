class EmbeddingStateManager:
    """Singleton class to manage state between transformer and embedding capture.
    Supports different methods of embedding capture: 'last', 'first', 'mean', 'max'
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.current_embedding = None
            cls._instance.current_method = "last"  # default method
        return cls._instance
    
    def set_method(self, method: str):
        """Set the method for capturing embeddings.
        
        Args:
            method (str): One of 'last', 'first', 'mean', 'max'
        """
        valid_methods = ['last', 'first', 'mean', 'max']
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.current_method = method
    
    def set_embedding_from_tensor(self, x):
        """Capture embedding from tensor based on current method.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, T, C]
        """
        if x.shape[1] <= 1:  # Skip if sequence length is 1 or less
            return
            
        if self.current_method == 'last':
            embedding = x[0, -1, :].detach().cpu()
        elif self.current_method == 'first':
            embedding = x[0, 0, :].detach().cpu()
        elif self.current_method == 'mean':
            embedding = x[0, :, :].mean(dim=0).detach().cpu()
        elif self.current_method == 'max':
            embedding = x[0, :, :].max(dim=0)[0].detach().cpu()
        else:
            raise ValueError(f"Unknown method: {self.current_method}")
            
        self.current_embedding = embedding
    
    def set_embedding(self, embedding):
        self.current_embedding = embedding
    
    def get_embedding(self):
        return self.current_embedding
    
    def clear_embedding(self):
        self.current_embedding = None

# Create a global instance
state_manager = EmbeddingStateManager()