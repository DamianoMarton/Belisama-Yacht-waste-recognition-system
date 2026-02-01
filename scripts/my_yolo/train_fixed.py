import time
import torch
from metrics_vect import compute_batch_performance_vect

def train(model, iterator, optimizer, criterion, device) -> float:
    """
    Trains the model for one epoch.

    Args:
        model: The neural network model to train.
        iterator: DataLoader for training data.
        optimizer: The optimizer for updating model weights.
        criterion: The loss function.
        device: The device to run the training on (CPU or GPU).

    Returns:
        float: The average loss over the epoch.
    """
    epoch_loss = 0.0
    model.train() # Set the model to training mode

    for (x, y) in iterator:
        x = x.to(device) 
        y = y.to(device)
        optimizer.zero_grad() # clear previous gradients
        y_pred = model(x) # forward pass
        loss = criterion(y_pred, y) # compute loss
        loss.backward() # backpropagation
        optimizer.step() # update weights
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device) -> tuple[float, float, float, float]:
    """
    Evaluates the model on the validation/test set.

    Args:
        model: The neural network model to evaluate.
        iterator: DataLoader for validation/test data.
        criterion: The loss function.
        device: The device to run the evaluation on (CPU or GPU).

    Returns:
        tuple: A tuple containing average loss, precision, recall, and F1 score.
    """
    epoch_loss = 0.0
    epoch_precision = 0.0
    epoch_recall = 0.0
    epoch_f1 = 0.0
    model.eval() # Set the model to evaluation mode
    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device) 
            y = y.to(device)
            y_pred = model(x) # forward pass
            loss = criterion(y_pred, y) # compute loss
            
            precision, recall, f1 = compute_batch_performance_vect(y_pred, y)
            
            epoch_loss += loss.item()
            epoch_precision += precision
            epoch_recall += recall
            epoch_f1 += f1

    avg_loss = epoch_loss / len(iterator)
    avg_precision = epoch_precision / len(iterator)
    avg_recall = epoch_recall / len(iterator)
    avg_f1 = epoch_f1 / len(iterator)

    return avg_loss, avg_precision, avg_recall, avg_f1

def model_training(n_epochs, 
                   model, 
                   train_iterator, valid_iterator, 
                   optimizer, criterion, 
                   device, scheduler=None, 
                   model_name='best_model.pt') -> tuple[list[float], list[float], list[float]]:
    """
    Trains the model for a specified number of epochs, evaluating on validation data.

    Args:
        n_epochs: Number of epochs to train.
        model: The neural network model to train.
        train_iterator: DataLoader for training data.
        valid_iterator: DataLoader for validation data.
        optimizer: The optimizer for updating model weights.
        criterion: The loss function.
        device: The device to run the training on (CPU or GPU).
        scheduler: Learning rate scheduler (optional).
        model_name: Filename to save the best model.

    Returns:
        tuple: Lists of training losses, validation losses, and validation F1 scores per epoch.
    """
    best_valid_loss = float('inf')
    train_losses, valid_losses, valid_f1s = [], [], []

    for epoch in range(n_epochs):
        start_time = time.time()
        
        train_loss = train(
            model, 
            train_iterator, 
            optimizer, 
            criterion, 
            device
        )
        valid_loss, valid_precision, valid_recall, valid_f1 = evaluate(
            model, 
            valid_iterator, 
            criterion, 
            device
        )
        
        if scheduler: # scheduler to adjust learning rate based on validation loss
            scheduler.step(valid_loss)

        end_time = time.time()

        print(f"\nEpoch: {epoch+1}/{n_epochs} | Time: {end_time-start_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {valid_loss:.4f} | Val Precision: {valid_precision * 100:.2f}% | Val Recall: {valid_recall * 100:.2f}% | Val F1: {valid_f1 * 100:.2f}%")
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_name)
            print(f"New best model saved: (Loss: {valid_loss:.4f})")

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        valid_f1s.append(valid_f1)

    return train_losses, valid_losses, valid_f1s