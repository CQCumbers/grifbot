from textgenrnn import textgenrnn

# load model
model = textgenrnn(
    weights_path='model_weights.hdf5',
    vocab_path='model_vocab.json',
    config_path='model_config.json',
)

# generate episodes
prefix = '\n\nRed vs. Blue Season 17\nEpiso'

model.generate_to_file(
    'gen_script.txt',
    prefix=prefix, n=1,
    max_gen_length=15000
)
