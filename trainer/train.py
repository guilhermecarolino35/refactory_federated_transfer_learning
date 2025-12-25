import torch

def train(net, trainloader,device, epochs: int, verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    
    net.train()
    for epoch in range(epochs):
        
        correct, total, epoch_loss = 0, 0, 0.0

        for batch in trainloader:
            images = batch["img"].to(device)
            labels =  batch["label"].to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Metrics
            batch_size = labels.size(0)
            epoch_loss += loss.item() * batch_size #soma ponderada
            
            total += batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()


        
        epoch_loss /= total #media por amostra
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss:.4f}, accuracy {epoch_acc:.4f}")

