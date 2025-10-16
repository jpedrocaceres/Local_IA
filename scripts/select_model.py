#!/usr/bin/env python3
"""
Interactive script to select which model to use
Saves selection to model_config.json
"""

import json
import os
import sys

def load_config():
    """Load current model configuration"""
    config_path = "model_config.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {
        "selected_model": "llama3.1",
        "models": {
            "llama3.1": {
                "name": "Llama 3.1 8B",
                "type": "llama",
                "description": "Meta's Llama 3.1 8B Instruct model for general chat and instruction following",
                "size": "~16GB (FP16)",
                "min_vram": "8GB",
                "recommended_vram": "16GB"
            },
            "trm": {
                "name": "TRM (Tiny Recursive Models)",
                "type": "trm",
                "description": "Samsung's 7M parameter recursive model for reasoning tasks (ARC-AGI, Sudoku, Mazes)",
                "size": "~28MB",
                "min_vram": "2GB",
                "recommended_vram": "4GB"
            }
        }
    }

def save_config(config):
    """Save model configuration"""
    with open("model_config.json", 'w') as f:
        json.dump(config, f, indent=2)

def display_models(config):
    """Display available models"""
    print("\n" + "=" * 70)
    print("Available Models:")
    print("=" * 70)

    models = config.get("models", {})
    current = config.get("selected_model", "llama3.1")

    for idx, (key, model_info) in enumerate(models.items(), 1):
        is_current = " (CURRENT)" if key == current else ""
        print(f"\n[{idx}] {model_info['name']}{is_current}")
        print(f"    Type: {model_info['type']}")
        print(f"    Description: {model_info['description']}")
        print(f"    Size: {model_info['size']}")
        print(f"    Min VRAM: {model_info['min_vram']}")
        print(f"    Recommended VRAM: {model_info['recommended_vram']}")

    print("\n" + "=" * 70)

def select_model():
    """Interactive model selection"""
    print("=" * 70)
    print("Model Selection Tool")
    print("=" * 70)

    # Load current configuration
    config = load_config()

    # Display available models
    display_models(config)

    models_list = list(config["models"].keys())

    while True:
        print(f"\nCurrent model: {config['selected_model']}")
        print("\nOptions:")
        for idx, key in enumerate(models_list, 1):
            print(f"  {idx}. Select {config['models'][key]['name']}")
        print("  q. Quit (no changes)")

        choice = input("\nEnter your choice: ").strip().lower()

        if choice == 'q':
            print("No changes made. Exiting...")
            return

        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(models_list):
                selected_key = models_list[choice_idx]
                selected_model = config["models"][selected_key]

                print(f"\n✓ Selected: {selected_model['name']}")
                print(f"  Description: {selected_model['description']}")

                # Confirm selection
                confirm = input("\nConfirm selection? (y/n): ").strip().lower()
                if confirm == 'y':
                    config["selected_model"] = selected_key
                    save_config(config)
                    print(f"\n✓ Model configuration saved!")
                    print(f"  Selected model: {selected_model['name']}")

                    # Check if model is ready
                    print("\nModel Status:")
                    if selected_key == "llama3.1":
                        print("  • Run 'python download_model.py' to download Llama 3.1")
                        print("  • Or use 'ollama pull llama3.1:8b' for Ollama")
                    elif selected_key == "trm":
                        print("  • Run 'python setup_trm.py' to install TRM")
                        print("  • Train model following TRM repository instructions")

                    print("\nTo start the server with selected model:")
                    print("  python main.py")
                    return
                else:
                    print("Selection cancelled.")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'q'.")

def main():
    """Main entry point"""
    try:
        select_model()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
