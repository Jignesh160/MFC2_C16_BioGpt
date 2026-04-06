import json
from torch.utils.data import Dataset, DataLoader

class MedicalDataset(Dataset):  # Changed from BiomedicalDataset
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loading dataset for MedBot training from: {data_path}")  # Updated message
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Handle BioASQ format
            self.data = []
            print(f"Processing {len(data['questions'])} questions...")
            
            for i, question in enumerate(data['questions'], 1):
                text = f"Question: {question['body']} Answer: {question['ideal_answer'][0] if isinstance(question['ideal_answer'], list) else question['ideal_answer']}"
                self.data.append({'text': text})
                if i % 100 == 0:
                    print(f"Processed {i} questions...")
            
            print(f"Dataset loading complete. Total samples: {len(self.data)}")
            print(f"Maximum sequence length: {max_length}")

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


# Update the test section at the end of file
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Initialize tokenizer for testing
    tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")  # Keep this as it's the model identifier
    
    # Test dataset loading
    data_path = r"C:\Users\shrav\Documents\Documents\medbot\src1\BioASQ-training11b\BioASQ-training11b\training11b.json"
    dataset = MedicalDataset(data_path, tokenizer)  # Updated class name
    
    # Test sample access
    print("\nTesting MedBot dataset access:")  # Updated message
    sample = dataset[0]
    print(f"Sample input_ids shape: {sample['input_ids'].shape}")
    print(f"Sample attention_mask shape: {sample['attention_mask'].shape}")
    
    # Test dataloader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print("\nTesting MedBot DataLoader:")  # Updated message
    batch = next(iter(dataloader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")