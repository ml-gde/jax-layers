from flax import nnx

from jaxgarden import T5Config, T5ForCausalLM, Tokenizer

if __name__ == "__main__":
    config = T5Config()
    model = T5ForCausalLM(config, rngs=nnx.Rngs(0))
    model_id = "google-t5/t5-base"

    # download checkpoint from HuggingFace Hub
    model.from_hf(model_id, force_download=True)

    tokenizer = Tokenizer.from_pretrained(model_id)

    text = "The meaning of life is"
    model_inputs = tokenizer.encode(text)
    output = model.generate(**model_inputs, max_length=20, do_sample=True)
    output_text = tokenizer.decode(output)

    print(output, output.shape)
    print(output_text)
