from flax import nnx

from jaxgarden import Gemma2Config, Gemma2ForCausalLM, Tokenizer

# HF repo id of the Gemma variant that you want to use
model_id = "google/gemma-2-2b-it"

# initialize the Gemma architecture
config = Gemma2Config()
model = Gemma2ForCausalLM(config, rngs=nnx.Rngs(0))

# This is a one-liner to download HF checkpoint from HuggingFace Hub,
# convert it to jaxgarden format,
# save it in an Orbax checkpoint,
# and then remove the HF checkpoint.
model.from_hf(model_id, force_download=True)

# this works just like `transformers.AutoTokenizer`,
# but without the dependency of the whole `transformers` library.
# Instead, we simply extend `tokenizers` package and add some cnvenience code for JAX.
tokenizer = Tokenizer.from_pretrained(model_id)

text = "The meaning of life is"
model_inputs = tokenizer.encode(text)
output = model.generate(**model_inputs, max_length=20, do_sample=True)
output_text = tokenizer.decode(output)
print(output_text)
