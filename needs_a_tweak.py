import modal

custom_image = (
    modal.Image.debian_slim(python_version="3.11")
)

app = modal.App("hello-modal")

@app.function(image=custom_image)
def say_something():
    message = # oops, I forgot to write a message
    return message

@app.local_entrypoint()
def main():
    print(say_something.remote())