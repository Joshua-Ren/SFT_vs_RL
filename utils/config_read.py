import yaml
from pathlib import Path

def load_config(config_path="config.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Convert null to None for Python compatibility
    def convert_none(obj):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: convert_none(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_none(item) for item in obj]
        return obj
    
    return convert_none(config)

# Usage in your training script
if __name__ == "__main__":
    # Load configuration
    config = load_config("/home/joshua52/projects/def-dsuth/joshua52/SFT_vs_RL/configs/train_basic.yaml")
    
    # You can now access configuration values
    print(f"Project: {config['project_name']}")
    print(f"Model: {config['model']['name']}")
    print(f"Learning rate: {config['training']['learning_rate']}")