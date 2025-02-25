import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from UNET_Bird_Segmentation import UNET
from dataset import train_loader, val_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for images, masks in train_loop:
            # Move data to device
            images = images.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.unsqueeze(1))  # Add channel dimension to masks
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update training loss
            train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks.unsqueeze(1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved! Best validation loss: {best_val_loss:.4f}')
        
        print('-' * 60)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Initialize model
    model = UNET(in_channels=3, out_channels=1).to(device)
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training parameters
    num_epochs = 100
    
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        device=device
    )

if __name__ == '__main__':
    main()