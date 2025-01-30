import modal

# Create an image and use Hugging Face's transformers library to cache the model
image = modal.Image.debian_slim(python_version="3.12") \
.pip_install(
    "transformers",
    "sentencepiece",
    "torch",
    "datasets") \
    .run_commands('python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; model_name = \'t5-small\'; T5Tokenizer.from_pretrained(model_name); T5ForConditionalGeneration.from_pretrained(model_name) "')


app = modal.App("T5-Small-Summarizer")

@app.function(image=image)
def summarize(text: str):
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024,
                              truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=40, length_penalty=2.0,
                                 num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

@app.local_entrypoint()
def main():
    text = """ 
Text of Lincoln's Speech
(Bliss copy)
Delivered at the dedication of the Soldiers' National Cemetery at Gettysburg, Pennsylvania.
Four score and seven years ago our fathers brought forth, on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.
Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.
But, in a larger sense, we can not dedicate, we can not consecrate, we can not hallow this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom—and that government of the people, by the people, for the people, shall not perish from the earth.

Abraham Lincoln

November 19, 1863.
"""
    # run the function locally
    print(summarize.remote(text))
