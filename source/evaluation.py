import torch
from torch.utils.data import DataLoader

from source.training_loop import collate_fn


def evaluate_model(model, dataset, loss_fn, batch_size, device="cuda"):

    model.to(device)
    model.eval()
    total_loss = 0.0
    total_samples = 0

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    with torch.no_grad():  #
        for speech_spec, melody_contour, melody_spec in dataloader:

            speech_spec = speech_spec.to(device)
            melody_contour = melody_contour.to(device)
            melody_spec = melody_spec.to(device)

            # Forward pass
            predictions = model(speech_spec, melody_contour)

            # Batch loss
            loss = loss_fn(predictions, melody_spec)
            batch_size = speech_spec.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    # Average loss
    loss_total = total_loss / total_samples
    return loss_total
