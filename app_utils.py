import yaml
from PIL import Image
import io
import base64

def load_yaml(file_path):
    """Load a YAML file and return its content."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)
    
def pil_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
    