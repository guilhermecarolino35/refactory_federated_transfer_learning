import torch

def test(net, testloader,device):

    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total = 0, 0
    total_loss = 0.0 
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size  # soma ponderada

            preds= outputs.argmax(dim=1)
            total += batch_size
            correct += (preds == labels).sum().item()
    
    avg_loss = total_loss / total 
    accuracy = correct / total
    return avg_loss, accuracy