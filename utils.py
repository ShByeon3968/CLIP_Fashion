import yaml

def load_yaml(file_path):
    """Load a YAML file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
    
    