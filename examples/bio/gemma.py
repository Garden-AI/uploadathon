from pathlib import Path
import modal

# Create a Volume, or retrieve it if it exists
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

@app.cls(
    image=gemma_image, 
    volumes={MODEL_DIR: volume},
    gpu="T4",
    enable_memory_snapshot=True
)
class TxGemmaPredictor:
    MODEL_VARIANT = "2b-predict"
    model_id = "google/txgemma-2b-predict"
    
    @modal.enter(snap=True)
    def load_model_to_cpu(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load tokenizer
        model_path = MODEL_DIR / self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Store only the configuration and model identifier during snapshot
        # We'll avoid loading the full model weights in this phase
        self.model_path = model_path
        self.config = {"device_map": "auto", "torch_dtype": torch.float16}
        
        # Load a minimal placeholder model for the snapshot
        # This avoids device transition issues later
        print("Preparing model configuration for snapshot...")
        
        # We're not actually loading the full model, just setting up for later
        # This prevents the device mismatch issues with KV cache
    
    @modal.enter(snap=False)
    def load_model_to_gpu(self):
        import torch
        from transformers import AutoModelForCausalLM
        
        # Load the model directly with proper GPU configuration after snapshot
        print("Loading model to GPU after snapshot restoration...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto",  # Let transformers handle device mapping
            torch_dtype=torch.float16,  # Use half precision for GPU
            local_files_only=True
        )
        print("Model loaded to GPU after snapshot restoration")
    
    @modal.method()
    def predict(self, prompt: str):
        import torch
        
        # Set eval mode
        self.model.eval()
        
        # Tokenize input and ensure it's on the correct device
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,
                use_cache=True,
                do_sample=False
            )
        
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return response

@app.function(
    volumes={MODEL_DIR: volume},
    image=gemma_image,
)
def download_model(
    repo_id: str="google/txgemma-2b-predict",
    revision: str=None,
    ):
    # (keep your existing download_model function)
    import os
    token = # token here
    os.environ["HF_TOKEN"] = token
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    
    repo_dir = MODEL_DIR / repo_id
    snapshot_download(repo_id=repo_id, local_dir=repo_dir, token=token)
    print(f"Model downloaded to {repo_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(repo_id, token=token)
    tokenizer.save_pretrained(repo_dir)
    print(f"Tokenizer downloaded to {repo_dir}")
    
    return "Model and tokenizer downloaded"

@app.local_entrypoint()
def foo():
    prompt = """Instructions: Answer the following question about drug properties.
Context: As a membrane separating circulating blood and brain extracellular fluid, the blood-brain barrier (BBB) is the protection layer that blocks most foreign drugs. Thus the ability of a drug to penetrate the barrier to deliver to the site of action forms a crucial challenge in development of drugs for central nervous system.
Question: Given a drug SMILES string, predict whether it
(A) does not cross the BBB (B) crosses the BBB
Drug SMILES: CN1C(=O)CN=C(C2=CCCCC2)c2cc(Cl)ccc21
"""
    # print(download_model.remote())
    
    # Use the class-based approach for prediction
    # predictor = TxGemmaPredictor()
    print(TxGemmaPredictor().predict.remote(prompt))