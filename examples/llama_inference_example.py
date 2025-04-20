from jaxgarden import LlamaConfig, LlamaForCausalLM, Tokenizer
from flax import nnx

if __name__ == "__main__":
    # initialize a config object (with defaults for 1B varient)
    # other varients to be added.
    config = LlamaConfig()
    model = LlamaForCausalLM(config, rngs=nnx.Rngs(0))
    model_id = "meta-llama/Llama-3.2-1B"

    # this will download HF checkpoint from HuggingFace Hub,
    # convert it to jaxgarden format,
    # save it in an Orbax checkpoint,
    # and then remove the HF checkpoint.
    # If you didn't set your HF token globally,
    # you may need to pass your token as an argument to this method.
    model.from_hf(model_id, force_download=True)

    # this works just like `transformers.AutoTokenizer`,
    # but without the dependency of the whole `transformers` library.
    # Instead, we simply extend `tokenizers` package and add some cnvenience code for JAX.
    tokenizer = Tokenizer.from_pretrained(model_id)
    
    text = "The meaning of life is"
    model_inputs = tokenizer.encode(text)
    output = model.generate(**model_inputs, max_length=20, do_sample=True)
    output_text = tokenizer.decode(output)
    print(output, output.shape)
    print(output_text)
