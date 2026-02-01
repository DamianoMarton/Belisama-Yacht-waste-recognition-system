import time
import torch
from model import MyYolo
from loss import YoloLoss
from dataset import MyYoloDataset
from metrics_vect import compute_batch_performance_vect

train_dataset = MyYoloDataset("./dataset/dataset", split='train')
val_dataset = MyYoloDataset("./dataset/dataset", split='val')

num_classes = train_dataset.num_classes

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

myyolo_model = MyYolo(num_classes)
criterion = YoloLoss(num_classes)
optimizer = torch.optim.Adam(myyolo_model.parameters(), lr=1e-4)
myyolo_model.to(device)
criterion.to(device)

def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0.0
    model.train()
    for (x,y) in iterator:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss/len(iterator)

def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_f1 = 0
    model.eval()
    with torch.no_grad():
        for (x, y) in iterator:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            
            # Now returns a scalar (F1 score)
            f1 = compute_batch_performance_vect(y_pred, y)
            
            epoch_loss += loss.item()
            epoch_f1 += f1

    return epoch_loss / len(iterator), epoch_f1 / len(iterator)

def model_training(n_epochs, model, 
                   train_iterator, valid_iterator, 
                   optimizer, criterion, 
                   device, 
                   model_name='best_model.pt'):

    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    valid_accs = []

    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = train(model, train_iterator, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model, valid_iterator, criterion, device)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name)
        end_time = time.time()

        print(f"\nEpoch: {epoch+1}/{n_epochs} -- Epoch Time: {end_time-start_time:.2f} s")
        print("---------------------------------")
        print(f"Train -- Loss: {train_loss:.3f}")
        print(f"Val -- Loss: {valid_loss:.3f}, F1: {valid_acc * 100:.2f}%")

        # Save
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

    return train_losses, valid_losses, valid_accs
