import torch

def train(epoch, model, train_loader, criterion, optimizer, device):
    print("Funzione train avviata!", flush=True)
    running_loss = 0.0
    correct = 0
    total = 0

    print("Numero batch:", len(train_loader), flush=True)


    for batch_idx, (inputs, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}", flush=True)
        inputs, targets = inputs.to(device), targets.to(device)

        # todo...
        optimizer.zero_grad()     #azzera i gradienti accumulati
        outputs = model(inputs)   # Forward pass
        loss = criterion(outputs, targets)    # Calcolo loss
        loss.backward()           # Backpropagation: calcola i gradienti
        optimizer.step()          # Aggiorno i pesi

        running_loss += loss.item() # estrae il valore numerico dal tensore loss e lo somma a running_loss
        _, predicted = outputs.max(1) # per ogni immagine nel batch, scegli la classe con punteggio più alto come predizione.
        total += targets.size(0) # conteggio cumulativo di quante immagini sono state viste finora.
        correct += predicted.eq(targets).sum().item() #numero totale di immagini correttamente classificate nell’epoch

    train_loss = running_loss / len(train_loader) #ottiene la loss media
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%', flush=True)

    # Validation loop
def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct, total = 0, 0

    with torch.no_grad():   # disabilita il calcolo dei gradienti (risparmia memoria e tempo)
        for inputs, targets in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # todo...
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Stats
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy