import modal

# Bring in dependencies and download the pytorch model from figshare
image = modal.Image.debian_slim(python_version="3.12").apt_install("wget") \
.pip_install(
    "colorama==0.4.6",
    "defusedxml==0.7.1",
    "numpy<2",
    "nvidia-ml-py==12.560.30",
    "pandas==2.2.3",
    "torch==2.2.2",
    "torchvision==0.17.2",
    "requests==2.32.3",
) \
    .run_commands("wget -O model.pth https://figshare.com/ndownloader/files/51767333", "pwd", "ls -lht")

app = modal.App("RetrogradeThawSlumps")

@app.function(image=image)
def identify_rts(image_url: str):
    import requests
    from torchvision.io import read_image
    import tempfile
    import torch

    def preprocess(image):
        from torchvision.io import read_image
        from torchvision.transforms import v2 as T
        transforms = []

        transforms.append(T.ToDtype(torch.float, scale=True))
        transforms.append(T.ToPureTensor())
        transform = T.Compose(transforms)

        image = (255.0 * (image - image.min()) / (image.max() - image.min())).to(
            torch.uint8
        )
        image = image[:3, ...]
        transformed_image = transform(image)

        x = torch.unsqueeze(transformed_image, 0)

        return x  # [:3, ...]

    def filter_predictions(pred, score_threshold=0.5):
        keep = pred["scores"] > score_threshold
        return {k: v[keep] for k, v in pred.items()}

    model = torch.load("/model.pth", map_location=torch.device('cpu'))

    response = requests.get(image_url)
    with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp_file:
        # Write the content to the temporary file
        tmp_file.write(response.content)
        tmp_file_path = tmp_file.name
        image = read_image(tmp_file_path)

    scaled_tensor = preprocess(image)
    with torch.no_grad():
        output = model(scaled_tensor)

    return filter_predictions(output[0])


@app.local_entrypoint()
def main():
    # run the function locally
    print(identify_rts.remote("https://github.com/cyber2a/Cyber2A-RTS-ToyModel/blob/main/data/images/valtest_nitze_008.jpg?raw=true"))
