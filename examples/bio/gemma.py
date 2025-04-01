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

with gemma_image.imports():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

@app.cls(
    image=gemma_image, 
    volumes={MODEL_DIR: volume},
    gpu="T4",
    enable_memory_snapshot=True
)
class TxGemmaPredictor:
    model_id = "google/txgemma-2b-predict"
    
    @modal.enter(snap=True)
    def load_model_to_cpu(self):
        """Load model weights to CPU memory only during snapshot creation"""

        # Load tokenizer
        model_path = MODEL_DIR / self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Load full model to CPU memory during snapshot
        print("Loading model to CPU for snapshot...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",  # Force CPU loading
            local_files_only=True
        )
        print("Model loaded to CPU for snapshot")
    
    @modal.enter(snap=False)
    def setup_gpu(self):
        """Move model to GPU after snapshot restoration"""
        
        print("Moving model to GPU after snapshot restoration...")
        # Simply transfer the entire model to GPU
        self.model = self.model.to("cuda")
        print("Model moved to GPU after snapshot restoration")
    
    @modal.method()
    def predict(self, prompt: str):
        """Generate predictions with the model"""        

        # Ensure model is in eval mode
        self.model.eval()
        
        # Process inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Disable KV cache to avoid device mismatch issues
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,
                use_cache=False,  # Disable KV cache to avoid device issues
                do_sample=False
            )
        
        # Return the decoded output
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

@app.cls(
    image=gemma_image, 
    volumes={MODEL_DIR: volume},
    gpu="T4",
    enable_memory_snapshot=True
)
class TxGemma9b:
    MODEL_VARIANT = "2b-predict"
    model_id = "google/txgemma-2b-predict"
    
    @modal.enter(snap=True)
    def load_model_to_cpu(self):
        """Load model weights to CPU memory only during snapshot creation"""

        # Load tokenizer
        model_path = MODEL_DIR / self.model_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        
        # Load full model to CPU memory during snapshot
        print("Loading model to CPU for snapshot...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",  # Force CPU loading
            local_files_only=True
        )
        print("Model loaded to CPU for snapshot")
    
    @modal.enter(snap=False)
    def setup_gpu(self):
        """Move model to GPU after snapshot restoration"""
        
        print("Moving model to GPU after snapshot restoration...")
        # Simply transfer the entire model to GPU
        self.model = self.model.to("cuda")
        print("Model moved to GPU after snapshot restoration")
    
    @modal.method()
    def predict(self, prompt: str):
        """Generate predictions with the model"""        

        # Ensure model is in eval mode
        self.model.eval()
        
        # Process inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        
        # Disable KV cache to avoid device mismatch issues
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=8,
                use_cache=False,  # Disable KV cache to avoid device issues
                do_sample=False
            )
        
        # Return the decoded output
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

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
    token = # insert token ... 
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
    print(download_model.remote("google/txgemma-9b-predict"))
    print(download_model.remote("google/txgemma-9b-chat"))
    # print(TxGemmaPredictor().predict.remote(prompt))