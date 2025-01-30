# T5-Small Text Summarizer
This uses the t5-small model to summarize text. The model is cached in the image and 
the function is ready to be registered as a garden

## Testing and Debugging
You will need the modal SDK to test the function. You can install it with pip:
```bash
pip install modal
```

Then you can test the function with the modal runtime:
```bash
modal run summarize.py
```

## Publishing to Garden
Once you have tested the function you can publish it to garden with the garden CLI.
First install garden with:
```bash
pip install garden-ai
```

Then you can publish the function with:
```
```bash

