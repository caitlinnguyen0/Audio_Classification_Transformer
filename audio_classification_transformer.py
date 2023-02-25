import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchvision.transforms as transforms
import torchaudio.transforms as ttransforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset

class AudioDataset(Dataset):
    def __init__(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        audio_file = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(audio_file)
        label = self.labels[idx]
        if self.transform:
            spectrogram = self.transform(waveform)
        return spectrogram, label

class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.transformer = nn.Transformer(d_model=128, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, dropout=0.1)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = self.transformer(x, x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

def train(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def test(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def predict(model, audio_file):
    transform = nn.Sequential(
        ttransforms.MelSpectrogram(sample_rate=44100, n_fft=2048, hop_length=512),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    )
    model.eval()
    with torch.no_grad():
        spectrogram = transform(audio_file).unsqueeze(0)
        outputs = model(spectrogram)
        _, preds = torch.max(outputs, 1)
    return preds.item()

if __name__ == '__main__':
    file_list = ['gunshot1.wav', 'gunshot2.wav', 'not_gunshot1.wav', 'not_gunshot2.wav']
    labels = [0, 0, 1, 1] # 0 for "gunshot", 1 for "not_gunshot"
dataset = AudioDataset(file_list, labels, transform=transforms.Compose([
    ttransforms.MelSpectrogram(sample_rate=44100, n_fft=2048, hop_length=512),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
]))
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
model = AudioClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10
for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_dataloader, criterion, optimizer)
    test_loss, test_acc = test(model, test_dataloader, criterion)
    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Test Loss: {:.4f}, Test Acc: {:.4f}'
          .format(epoch+1, num_epochs, train_loss, train_acc, test_loss, test_acc))
audio_file = 'gunshot_test.wav'
predicted_label = predict(model, audio_file)
print('Predicted label:', 'gunshot' if predicted_label == 0 else 'not_gunshot')


