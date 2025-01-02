import torch

# Define accuracy function
def accuracy_fn(y_pred: torch.Tensor, y_true: torch.Tensor):
    return (y_pred == y_true).sum().item()/len(y_true)


# Training function
def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.long().to(device)

        # 1. Forward pass
        y_logits = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_logits, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
        train_acc += accuracy_fn(y_pred_class, y)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


# Validation function
def val_step(model: torch.nn.Module, 
             dataloader: torch.utils.data.DataLoader, 
             loss_fn: torch.nn.Module, 
             accuracy_fn,
             device):
    # Put the model in eval mode
    model.eval()

    # Setup validation loss and train accuracy values
    val_loss, val_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through each batch
        for batch, (X,y) in enumerate(dataloader):
            # Send the data to target device
            X,y = X.to(device), y.long().to(device)

            # Forward pass
            y_logits = model(X)

            # Calc and accumulate validation loss
            loss = loss_fn(y_logits, y)
            val_loss += loss.item()
            
            # Calc and accumulate val loss
            y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            val_acc += accuracy_fn(y_pred, y)
            

        # Adjust metrics to get average loss and accuracy per batch 
        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)
        return val_loss, val_acc

# Model evaluation function
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device):
    """Returns a dictionary containing the results of model predicting on data_loader.
         - model_name
         - loss
         - accuracy
    """
    # Setup loss and acc
    loss, acc = 0, 0
    
    model.eval()
    # Turn on inference mode
    with torch.inference_mode():
        # Loop through each batch of data
        for X, y in data_loader:
            # Send the data to target device
            X,y = X.to(device), y.long().to(device)  # Cross Entropy expects y to be a long integer
            
            # Make predictions with the model
            y_logits = model(X)
            y_pred = torch.argmax(torch.softmax(y_logits, dim=1), dim=1)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_logits, y)
            acc += accuracy_fn(y_pred, y) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, 
            "model_loss": loss.item(),
            "model_acc": acc}
