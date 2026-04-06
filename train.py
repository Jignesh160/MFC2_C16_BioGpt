from model import MedBotModel
from data_processor import MedicalDataset
from torch.utils.data import DataLoader
import torch
import os

def train_model(data_path, epochs=5, batch_size=2):
    # Initialize model
    medbot = MedBotModel()
    
    # Create dataset
    dataset = MedicalDataset(data_path, medbot.tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(medbot.model.parameters(), lr=2e-5)
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            input_ids = batch['input_ids'].to(medbot.device)
            attention_mask = batch['attention_mask'].to(medbot.device)
            
            optimizer.zero_grad()
            outputs = medbot.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(medbot.model.state_dict(), f"{checkpoint_dir}/medbot_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    data_path = r"C:\Users\shrav\Documents\Documents\medbot\src1\BioASQ-training11b\BioASQ-training11b\training11b.json"
    train_model(data_path)