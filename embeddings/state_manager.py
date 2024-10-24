class EmbeddingStateManager:
    """Singleton class to manage state between transformer and embedding capture."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.current_embedding = None
        return cls._instance
    
    def set_embedding(self, embedding):
        self.current_embedding = embedding
    
    def get_embedding(self):
        return self.current_embedding
    
    def clear_embedding(self):
        self.current_embedding = None

# Create a global instance
state_manager = EmbeddingStateManager()