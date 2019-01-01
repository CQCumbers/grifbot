from textgenrnn import textgenrnn

# load model
model = textgenrnn(
    weights_path='model_weights.hdf5',
    vocab_path='model_vocab.json',
    config_path='model_config.json',
)

# generate episodes
temperature = [1.0, 0.5, 0.2, 0.2]   
prefix = '\n\nRed vs. Blue Season 17\nEpiso'

model.generate_to_file(
    'gen_script.txt',
    temperature=temperature,
    prefix=prefix, n=1,
    max_gen_length=15000
)
