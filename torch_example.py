import modal

torch_image = (
    modal.Image.from_registry("python:3.11-bullseye")
    .apt_install("git", "git-lfs")
    # .pip_install("scikit-learn==1.2.2", "pandas==2.1.2")
    .run_commands("pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121")
    .run_commands("git clone https://huggingface.co/willengler-uc/torch-example")
)

app = modal.App("torch-example")

# This is a model trained to predict the line y = 2x + 1
@app.function(image=torch_image)
def run_regression_model(input_floats):
    import sys
    import torch

    # Adds model definition to path (inside of model_definition.py)
    sys.path.append('/torch-example')

    from model_definition import SimpleModel
    loaded_model = SimpleModel()
    loaded_model.load_state_dict(torch.load('/torch-example/simple_model.pth', weights_only=True))
    loaded_model.eval()  # Set to evaluation mode

    input_tensor = torch.tensor([[x] for x in input_floats]).float()

    with torch.no_grad():
        prediction = loaded_model(input_tensor)
        return prediction.squeeze().tolist()


@app.local_entrypoint()
def test_regression_model():
    example_input = [1.0, 2.0]
    result = run_regression_model.remote(example_input)
    print(result)
    return result