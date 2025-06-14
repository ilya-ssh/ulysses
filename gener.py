import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from collections import Counter
from datasets import load_dataset
import json
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Используем устройство:", device)
#загрузить словарь
with open("word_index.json", "r", encoding="utf-8") as f:
    word_index = json.load(f)
word_index = { token: int(idx) for token, idx in word_index.items() }
with open("index_to_word.json", "r", encoding="utf-8") as f:
    tmp = json.load(f)
index_to_word = { int(idx): token for idx, token in tmp.items() }
print("Словарь word_index:")
print(word_index)
print("Обратный словарь index_to_word:")
print(index_to_word)
vocab_size = len(word_index) + 1
EMB_DIM = 100
HIDDEN_DIM = 150
class UlyssesModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super(UlyssesModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        lengths, perm_idx = lengths.sort(0, descending=True)
        embedded = embedded[perm_idx]
        packed_input = rnn_utils.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True)
        packed_output, (hidden, _) = self.lstm(packed_input)
        hidden = hidden.squeeze(0)

        _, unperm_idx = perm_idx.sort(0)
        hidden = hidden[unperm_idx]
        output = self.fc(hidden)
        return output

model = UlyssesModel(vocab_size, EMB_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(torch.load("ulysses_text_model.pt", map_location=device))
model.eval()

def generate_text(seed_text, next_words, word_index, index_to_word):
    model.eval()
    tokens = seed_text.split()
    with torch.no_grad():
        for _ in range(next_words):
            seq = [word_index.get(token, 0) for token in tokens]
            input_tensor = torch.tensor([seq], dtype=torch.long).to(device)
            length_tensor = torch.tensor([len(seq)], dtype=torch.long).to(device)
            output = model(input_tensor, length_tensor)
            predicted_index = output.argmax(dim=-1).item()
            predicted_word = index_to_word.get(predicted_index, "")
            tokens.append(predicted_word)
    return " ".join(tokens)

while True:
    seed_text = input("Первое слово предложения: ")
    if seed_text.lower() == 'esc':
        break
    generated = generate_text(seed_text, 100, word_index, index_to_word)
    print("Сгенерированный текст:")
    print(generated)
