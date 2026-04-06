from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class MedBotModel:
    def __init__(self):
        self.device = torch.device("cpu")  # Force CPU usage
        self.model_name = "microsoft/biogpt"  # Keep this as it's the actual model name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
    def generate_response(self, input_text, max_length=200):
        # Detect question type and create appropriate prompt
        question_type = self._detect_question_type(input_text.lower())
        formatted_prompt = self._create_prompt(input_text, question_type)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True
        ).to(self.device)
        
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=50,  # Increased for better definitions
            num_return_sequences=1,
            num_beams=5,    # Increased for more accurate responses
            no_repeat_ngram_size=2,
            do_sample=True,
            top_k=50,
            temperature=0.4, # Adjusted for better balance
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True,
            repetition_penalty=1.5  # Increased to avoid repetition
        )
        
        # Process the response using the appropriate cleaner
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        cleaned_response = self._clean_response(response, question_type)
        
        return cleaned_response
    
    def _detect_question_type(self, question):
        if 'what is' in question.lower() or 'define' in question.lower() or 'meaning of' in question.lower():
            return 'definition'
        if any(word in question for word in ['symptom', 'sign']):
            return 'symptoms'
        elif any(word in question for word in ['treat', 'cure', 'manage']):
            return 'treatment'
        elif any(word in question for word in ['cause', 'risk', 'factor']):
            return 'causes'
        elif any(word in question for word in ['prevent', 'avoid']):
            return 'prevention'
        return 'general'
    
    def _create_prompt(self, question, q_type):
        # Extract the medical term/condition
        if q_type == 'definition':
            term = question.lower().replace("what is", "").replace("define", "").replace("meaning of", "").replace("?", "").strip()
            return (
                f"Provide a clear and concise medical/scientific definition of: {term}.\n"
                f"Definition: {term} is"
            )
        
        condition = question.replace("what are", "").replace("?", "").strip()
        
        prompts = {
            'symptoms': f"List only the classic and most common symptoms of {condition} according to medical guidelines.\nThe main symptoms include:",
            'treatment': f"List the standard medical treatments for {condition} according to clinical guidelines.\nThe main treatments include:",
            'causes': f"List the primary causes and risk factors of {condition} according to medical research.\nThe main causes include:",
            'prevention': f"List the most effective ways to prevent {condition} according to medical guidelines.\nThe prevention methods include:",
            'definition': f"Provide a clear medical definition of the term: {condition}.\nDefinition:",
            'general': f"Provide medical information about {condition} according to clinical guidelines.\nKey points include:"
        }
        return prompts.get(q_type, prompts['general'])
    
    def _clean_response(self, response, q_type):
        # Remove prompt and get only the relevant part
        split_phrases = {
            'symptoms': "include:",
            'treatment': "include:",
            'causes': "include:",
            'prevention': "include:",
            'definition': "Definition:",
            'general': "include:"
        }
        split_phrase = split_phrases.get(q_type, "include:")
        cleaned_response = response.split(split_phrase)[-1].strip()
        return cleaned_response.lstrip(',:. ')

# Update main section
if __name__ == "__main__":
    print("Initializing MedBot Assistant...")
    model = MedBotModel()
    
    print("\nWelcome to MedBot Assistant")
    print("Example questions:")
    print("- What are the symptoms of [condition]?")
    print("- What are the treatments for [condition]?")
    print("- What causes [condition]?")
    print("- What is [medical term]?")
    print("- Define [medical condition]")
    
    while True:
        print("\nEnter your medical question (or 'quit' to exit):")
        user_question = input("> ").strip()
        
        if user_question.lower() == 'quit':
            print("Thank you for using Medical Assistant!")
            break
            
        if user_question:
            print("\nGenerating response...")
            response = model.generate_response(user_question)
            print("\nResponse:", response)
        else:
            print("Please enter a valid question.")