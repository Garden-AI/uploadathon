import modal

custom_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==1.24.3")
)

app = modal.App("hello-garden")

@app.function(image=custom_image)
def say_hello():
    import numpy as np
    print(np.__version__)
    return "Hello, Garden!"
