from pathlib import Path


import modal

# Make image with transformers and such ...
# 1: Get model weights into a Volume
# 2: Load it and run

# create a Volume, or retrieve it if it exists
volume = modal.Volume.from_name("txgemma-weights-vol", create_if_missing=True)
MODEL_DIR = Path("/models")

gemma_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "git-lfs")
    .pip_install("huggingface_hub[hf_transfer]")  # install fast Rust download client
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # and enable it
    .pip_install("transformers", "torch")
    .pip_install("bitsandbytes", "accelerate", "transformers==4.48.3")
)

app = modal.App("txgemma-garden")

@app.function(
    image=gemma_image, 
    volumes={MODEL_DIR: volume}, 
    gpu="T4"
)
def predict(prompt: str):
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    MODEL_VARIANT = "2b-predict"
    additional_args = {}

    model_id = "google/txgemma-2b-predict"
    # Load both model and tokenizer from Volume
    model_path = MODEL_DIR / model_id
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        **additional_args,
    )

    # I'll need examples with good prompts in the notebook
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=8)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.function(
    volumes={MODEL_DIR: volume},  # "mount" the Volume, sharing it with your function
    image=gemma_image,  # only download dependencies needed here
)
def download_model(
    repo_id: str="google/txgemma-2b-predict",
    revision: str=None,  # include a revision to prevent surprises!
    ):
    import os
    token = # add token 
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    
    repo_dir = MODEL_DIR / repo_id
    # if not repo_dir.exists():
    # Download model weights
    snapshot_download(repo_id=repo_id, local_dir=repo_dir, token=token)
    print(f"Model downloaded to {repo_dir}")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
    tokenizer.save_pretrained(repo_dir)
    print(f"Tokenizer downloaded to {repo_dir}")
    
    return "Model and tokenizer downloaded"
    # else:
    #     return "Model and tokenizer already exist"

@app.local_entrypoint()
def foo():
    prompt = """Instructions: Answer the following question about drug properties.
Context: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system.
Question: Given a drug SMILES string, predict whether it
(A) does not cross the BBB (B) crosses the BBB
Drug SMILES: CN1C(=O)CN=C(C2=CCCCC2)c2cc(Cl)ccc21
"""
    # print(download_model.remote())
    print(predict.remote(prompt))