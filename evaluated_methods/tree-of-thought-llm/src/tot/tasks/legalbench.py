import re
import os
import json
import glob
import random
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.legalbench import *

class LegalBenchTask(Task):
    def __init__(self, file=None):
        super().__init__()
        
        # Set up the LegalBench data directory
        legalbench_dir = os.path.join(DATA_PATH, 'LegalBench')
        pp_dir = os.path.join(legalbench_dir, 'privacy_policy_qa')
        
        # Create directories if they don't exist
        os.makedirs(legalbench_dir, exist_ok=True)
        os.makedirs(pp_dir, exist_ok=True)
        
        # Check if we need to download the dataset
        if file is None:
            test_path = os.path.join(pp_dir, "test.jsonl")
            if not os.path.exists(test_path):
                file = self.download_legalbench_data(pp_dir)
            else:
                file = "test.jsonl"
        
        path = os.path.join(pp_dir, file)
        
        # If the file doesn't exist, download the data
        if not os.path.exists(path):
            print(f"File {path} not found. Downloading LegalBench dataset.")
            file = self.download_legalbench_data(pp_dir)
            path = os.path.join(pp_dir, file)
            
        # Load the data
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]
            
        print(f"Loaded {len(self.data)} examples from {path}")
        
        # Sample data check - print a few examples with their labels
        print("First 3 examples:")
        for i in range(min(3, len(self.data))):
            # Look for label in both 'label' and 'answer' fields
            label = self.data[i].get('label', self.data[i].get('answer', 'No label'))
            question = self.data[i].get('question', 'No question')[:30]
            clause = self.data[i].get('clause', self.data[i].get('text', 'No clause'))[:30]
            print(f"Item {i}: Q: {question}... Clause: {clause}... Label: {label}")
        
        self.value_cache = {}
        self.steps = 5
        self.stops = ['\n'] * 5
    
    def download_legalbench_data(self, output_dir):
        """Download the privacy_policy_qa dataset from HuggingFace"""
        try:
            from datasets import load_dataset
            
            print("Downloading LegalBench privacy_policy_qa dataset from HuggingFace...")
            dataset = load_dataset("nguha/legalbench", "privacy_policy_qa")
            
            # Save the train and test splits
            train_path = os.path.join(output_dir, "train.jsonl")
            test_path = os.path.join(output_dir, "test.jsonl")
            
            with open(train_path, 'w') as f:
                for item in dataset['train']:
                    f.write(json.dumps(item) + "\n")
                
            with open(test_path, 'w') as f:
                for item in dataset['test']:
                    f.write(json.dumps(item) + "\n")
                
            print(f"Downloaded dataset with {len(dataset['train'])} train and {len(dataset['test'])} test examples")
            return "test.jsonl"  # Return the filename to use
        except ImportError:
            print("Error: 'datasets' library not installed. Please install with: pip install datasets")
            print("Falling back to synthetic dataset.")
            return self._create_synthetic_dataset(output_dir)
        except Exception as e:
            print(f"Error downloading dataset: {str(e)}")
            print("Falling back to synthetic dataset.")
            return self._create_synthetic_dataset(output_dir)
    
    def _create_synthetic_dataset(self, output_dir):
        """Create a synthetic dataset with 200 examples"""
        synthetic_data = []
        
        # Templates to create varied data
        questions = [
            "Do you collect my personal information?",
            "Do you share my data with third parties?",
            "How long do you retain my data?",
            "Can I delete my account?",
            "Do you use encryption to protect my data?",
            "Do you use cookies on your website?",
            "How do you use my location data?",
            "Can I opt out of marketing communications?",
            "Do you collect data from children?",
            "What happens to my data if you're acquired by another company?"
        ]
        
        relevant_clauses = [
            "We collect personal information such as your name, email address, and browsing history to provide our services.",
            "We may share your personal information with third-party service providers who perform services on our behalf.",
            "We retain your personal information for as long as necessary to fulfill the purposes described in this Privacy Policy.",
            "You may request deletion of your account and personal information by contacting our support team.",
            "We use industry-standard encryption technologies to protect your personal information during transmission.",
            "Our website uses cookies to enhance your browsing experience and analyze website traffic.",
            "We collect and use location data to provide location-based services and advertising.",
            "You can opt out of receiving marketing communications by clicking the unsubscribe link in our emails.",
            "Our services are not intended for children under 13, but we may collect information with parental consent.",
            "If our company is acquired, your personal information may be transferred to the acquiring entity."
        ]
        
        irrelevant_clauses = [
            "Our website is hosted on secure servers located in the United States.",
            "This Privacy Policy was last updated on January 1, 2023.",
            "We may update this Privacy Policy from time to time.",
            "By using our service, you agree to our Terms of Service.",
            "Our company is headquartered in San Francisco, California.",
            "We have designated a Data Protection Officer to oversee our privacy program.",
            "You can contact us at privacy@example.com with any questions.",
            "Disputes will be resolved through arbitration in accordance with our Terms of Service.",
            "Our service may contain links to third-party websites.",
            "We reserve the right to modify or terminate our service at any time."
        ]
        
        # Generate 200 examples with explicit labels
        for i in range(200):
            # Select a question
            question = questions[i % len(questions)]
            
            # Decide if it should be relevant or irrelevant (alternating)
            if i % 2 == 0:
                # Relevant example
                clause = relevant_clauses[i % len(relevant_clauses)]
                label = "Relevant"
            else:
                # Irrelevant example
                clause = irrelevant_clauses[i % len(irrelevant_clauses)]
                label = "Irrelevant"
            
            # Add to dataset with explicit label
            synthetic_data.append({
                "question": question,
                "clause": clause,
                "label": label  # Explicit label
            })
        
        # Shuffle the data
        random.shuffle(synthetic_data)
        
        # Save to file
        output_path = os.path.join(output_dir, "synthetic_test.jsonl")
        with open(output_path, 'w') as f:
            for item in synthetic_data:
                f.write(json.dumps(item) + "\n")
                
        print(f"Created synthetic dataset with {len(synthetic_data)} examples at {output_path}")
        return "synthetic_test.jsonl"
    
    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> dict:
        """Return the question and clause as a dictionary to match prompt expectations"""
        question = self.data[idx].get('question', 'No question found')
        
        # Look for clause in both 'clause' and 'text' fields to support different dataset formats
        clause = self.data[idx].get('clause', self.data[idx].get('text', 'No clause found'))
        
        return {
            "question": question,
            "clause": clause
        }

    def test_output(self, idx: int, output: str):
        """Test if the output matches the expected answer"""
        # Get the correct answer from the data - check both 'label' and 'answer' fields
        correct_answer = self.data[idx].get('label', self.data[idx].get('answer', None))
        
        # If no label is found, default to irrelevant
        if not correct_answer:
            print(f"Warning: No label found for index {idx}, defaulting to 'Irrelevant'")
            correct_answer = "Irrelevant"
        
        # Extract the predicted answer from the output
        predicted_answer = self.extract_answer(output)
        
        # Compare and return result
        if predicted_answer:
            is_correct = predicted_answer.lower() == correct_answer.lower()
            return {'r': int(is_correct)}
        else:
            print(f"Warning: No predicted answer found in output for index {idx}")
            return {'r': 0}

    @staticmethod
    def extract_answer(output: str) -> str:
        """Extract the answer from the model's output"""
        # For binary classification (Relevant/Irrelevant)
        relevant_patterns = [
            r'####\s*(Relevant)', 
            r'The (?:final |correct )?answer is:?\s*(Relevant)',
            r'\b(Relevant)\b',
            r'The clause is (relevant)',
            r'This clause is (relevant)'
        ]
        
        irrelevant_patterns = [
            r'####\s*(Irrelevant)', 
            r'The (?:final |correct )?answer is:?\s*(Irrelevant)',
            r'\b(Irrelevant)\b',
            r'The clause is (irrelevant)',
            r'This clause is (irrelevant)'
        ]
        
        for pattern in relevant_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return "Relevant"
                
        for pattern in irrelevant_patterns:
            match = re.search(pattern, output, re.IGNORECASE)
            if match:
                return "Irrelevant"
        
        # Simple fallback
        if "relevant" in output.lower():
            return "Relevant"
        if "irrelevant" in output.lower():
            return "Irrelevant"
            
        return "Irrelevant"  # Default answer

    @staticmethod
    def standard_prompt_wrap(x: str, y: str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str='') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        return propose_prompt.format(input=x, partial_solution=y)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        return value_prompt.format(input=x, partial_solution=y)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # Simplified value calculation
        if "relevant" in y.lower() or "irrelevant" in y.lower():
            return 1.0
        return 0.5  # Default