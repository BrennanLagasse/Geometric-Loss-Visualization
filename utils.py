import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm.notebook import tqdm

def evaluate(model, loader, loss_fn, device, num_epochs=None):
    """ Basic evaluation script """

    model.eval()

    total_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for epoch, (images, labels) in enumerate(tqdm(loader)):

            if num_epochs is not None and epoch >= num_epochs:
                break

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

    acc = correct / total
    avg_loss = total_loss / total


    return acc, avg_loss

def model_to_vector(model):
    return parameters_to_vector(model.parameters()).detach()

def vector_to_model(vec, model):
    vector_to_parameters(vec, model.parameters())