import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import re
from collections import Counter
from datasets import load_dataset
from multiprocessing import freeze_support
import json
def tokenize(text):
    return re.findall(r'\w+|[^\w\s]', text.lower())

class UlyssesDataset(IterableDataset):
    def __init__(self, texts, word_index):
        self.texts = texts
        self.word_index = word_index

    def __iter__(self):
        for text in self.texts:
            tokens = tokenize(text)
            seq = [self.word_index.get(token, 0) for token in tokens]
            if len(seq) < 2:
                continue
            for i in range(1, len(seq)):
                yield (torch.tensor(seq[:i], dtype=torch.long),torch.tensor(seq[i], dtype=torch.long))

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    sequences_padded = rnn_utils.pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return sequences_padded, lengths, labels

class UlyssesModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim):
        super().__init__() #нициализируем базовый класс nn.Module
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0) #словарь в эмбед
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
        return self.fc(hidden)

def main():
    freeze_support()  #иначе ошибка на windows из-за распараллеливания dataloader
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Используем устройство:", device)
    #загружаем и токенизируем датасет
    dataset = load_dataset("bomjara/ulysses")
    texts = dataset["train"]["text"]
    #словарь и обратный словарь
    word_counts = Counter()
    for text in texts:
        word_counts.update(tokenize(text))
    word_index = {w: i for i, (w, _) in enumerate(word_counts.items(), start=1)}
    vocab_size = len(word_index) + 1
    index_to_word = {i: w for w, i in word_index.items()}
    with open("word_index.json", "w", encoding="utf-8") as f:
        json.dump(word_index, f, ensure_ascii=False, indent=2)
    with open("index_to_word.json", "w", encoding="utf-8") as f:
        json.dump({str(i): w for i, w in index_to_word.items()}, f, ensure_ascii=False, indent=2)
    #даталоадер с распараллеливанием
    BATCH_SIZE = 64
    dataset_pt = UlyssesDataset(texts, word_index)
    dataloader = DataLoader(dataset_pt,batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    #модель
    model = UlyssesModel(vocab_size, emb_dim=100, hidden_dim=150).to(device)
    print("Модель:")
    print(model)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    num_epochs = 50 #should be more probably, based on the quality of the model
    for epoch in range(num_epochs):
        model.train()
        total_loss = total_correct = total_samples = batch_count = 0
        for sequences, lengths, labels in dataloader:
            sequences = sequences.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(sequences, lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item()
            batch_count += 1
        avg_loss = total_loss / batch_count
        accuracy = total_correct / total_samples * 100
        print(f"Эпоха {epoch+1}/{num_epochs}, Потеря: {avg_loss:.4f}, Точность: {accuracy:.2f}%")
    def generate_text(seed_text, next_words):
        model.eval()
        tokens = tokenize(seed_text)
        with torch.no_grad():
            for _ in range(next_words):
                seq = [word_index.get(t, 0) for t in tokens]
                inp = torch.tensor([seq], dtype=torch.long).to(device)
                length = torch.tensor([len(seq)], dtype=torch.long).to(device)
                out = model(inp, length)
                idx = out.argmax(dim=1).item()
                tokens.append(index_to_word.get(idx, ""))
        return " ".join(tokens)
    print("Сгенерированный текст:")
    print(generate_text("night", 10))
    #cохраняем веса модели
    torch.save(model.state_dict(), 'ulysses_text_model.pt')

if __name__ == "__main__":
    main()
