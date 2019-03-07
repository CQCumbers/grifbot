import torch
from torch.nn import functional as F
from pytorch_pretrained_bert import GPT2Tokenizer, GPT2LMHeadModel, OpenAIAdam
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = tokenizer.encode("Simmons:Do you ever wonder why we're here?")

"""
output_model_file = "pytorch_model.bin"
model = GPT2LMHeadModel.from_pretrained('gpt2')
config = model.config
model_state_dict = torch.load(output_model_file)
model = GPT2LMHeadModel(config)
model.load_state_dict(model_state_dict)
model.to(device)
model.eval()
input, past = torch.tensor([text]), None

with torch.no_grad():
  for _ in range(50):
    logits, past = model(input.to(device), past=past)
    log_probs = F.softmax(logits[:, -1], dim=-1)
    input = torch.multinomial(log_probs, 1)
    text.append(input.item())

print(tokenizer.decode(text))
print(torch.cuda.memory_allocated(device))

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange

# load scripts from text file
file_name = 'scripts3.txt'
lines = open(file_name, 'r', encoding='utf8').read()
text = []
for line in lines.split('\n'):
  text.extend(tokenizer.encode(line + '\n'))

# split text into sequences
input_len = 128
data = []
for i in trange(len(text) - input_len):
  data.append(text[i:i + input_len])
data = torch.tensor(data)
data, targets = data[:-1], data[1:]

# turn sequences into batched data loader
batch_size = 8
train_data = TensorDataset(data, targets)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

# Prepare optimizer
num_epochs = 1
learning_rate = 6.25e-10
warmup_proportion = 0.002
max_grad_norm = 0.05
weight_decay = 0.01

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
  {'params': [p for n, p in param_optimizer if
    not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
  {'params': [p for n, p in param_optimizer if
    any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = len(train_data) * num_epochs // batch_size
optimizer = OpenAIAdam(optimizer_grouped_parameters,
                       lr=learning_rate,
                       warmup=warmup_proportion,
                       max_grad_norm=max_grad_norm,
                       weight_decay=weight_decay,
                       t_total=num_train_optimization_steps)

# Run training
nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
model.train()
for _ in range(int(num_epochs)):#trange(int(num_epochs), desc="Epoch"):
  tr_loss = 0
  nb_tr_steps = 0
  tqdm_bar = tqdm(train_dataloader, desc="Training")
  for step, batch in enumerate(tqdm_bar):
    input_ids, lm_labels = tuple(t.to(device) for t in batch)
    loss = model(input_ids, lm_labels=lm_labels)
    loss.backward()
    optimizer.step()
    tr_loss += loss.item()
    exp_average_loss = (loss.item() if exp_average_loss is None else
      0.7*exp_average_loss+0.3*loss.item())
    nb_tr_steps += 1
    tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(
      exp_average_loss, optimizer.get_lr()[0])

model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = "pytorch_model.bin"
config = model.config
torch.save(model_to_save.state_dict(), output_model_file)
"""

# Load a trained model that you have fine-tuned
output_model_file = "pytorch_model.bin"
model = GPT2LMHeadModel.from_pretrained('gpt2')
config = model.config
model_state_dict = torch.load(output_model_file)
model = GPT2LMHeadModel(config)
model.load_state_dict(model_state_dict)
model.to(device)

text = tokenizer.encode("Blood Gulch Chronicles Season 17\nEpisode 1: ")
model.eval()
input, past = torch.tensor([text]), None
for _ in range(128):
  logits, past = model(input.to(device), past=past)
  log_probs = F.softmax(logits[:, -1], dim=-1)
  input = torch.multinomial(log_probs, 1)
  text.append(input.item())
print(tokenizer.decode(text), end="")

with torch.no_grad():
  for _ in range(100):
    text = text[-64:]
    input, past = torch.tensor([text]), None
    for _ in range(128):
      logits, past = model(input.to(device), past=past)
      log_probs = F.softmax(logits[:, -1], dim=-1)
      input = torch.multinomial(log_probs, 1)
      text.append(input.item())
    print(tokenizer.decode(text[64:]), end="")

