import re
import os
import json
import glob
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.medqa import *

class MedQATask(Task):
    def __init__(self, file=None):
        super().__init__()
        
        # Set up the MedQA data directory
        medqa_dir = os.path.join(DATA_PATH, 'MedQA')
        os.makedirs(medqa_dir, exist_ok=True)
        
        # Try to find the specified file or any JSONL file
        if file is None:
            jsonl_files = glob.glob(os.path.join(medqa_dir, '*.jsonl'))
            
            if jsonl_files:
                file = os.path.basename(jsonl_files[0])
                print(f"Using existing MedQA file: {file}")
            else:
                print(f"No MedQA data found in {medqa_dir}. Using synthetic dataset.")
                file = self._create_medqa_dataset(medqa_dir)
        
        path = os.path.join(medqa_dir, file)
        
        # Load the data
        try:
            with open(path, 'r') as f:
                self.data = [json.loads(line) for line in f]
            
            print(f"Loaded {len(self.data)} examples from {path}")
            
            # Check for options in the first example
            if len(self.data) > 0:
                first_item = self.data[0]
                if not first_item.get('options') or len(first_item.get('options', {})) == 0:
                    print(f"WARNING: First item has no options: {first_item}")
            
            # Print a few examples for verification
            print("First 3 examples:")
            for i in range(min(3, len(self.data))):
                answer_idx = self.data[i].get('answer_idx', 'No answer')
                question = self.data[i].get('question', 'No question')[:50] + '...'
                num_options = len(self.data[i].get('options', {}))
                print(f"Item {i}: Q: {question} | Options: {num_options} | Answer: {answer_idx}")
        
        except Exception as e:
            print(f"Error loading MedQA data from {path}: {str(e)}")
            print("Creating synthetic dataset instead")
            file = self._create_medqa_dataset(medqa_dir)
            path = os.path.join(medqa_dir, file)
            with open(path, 'r') as f:
                self.data = [json.loads(line) for line in f]
        
        self.value_cache = {}
        self.steps = 5
        self.stops = ['\n'] * 5
    
    def _validate_and_fix_data(self):
        """Ensure all data entries have proper options and answers"""
        valid_answers = set(['A', 'B', 'C', 'D', 'E'])
        
        for idx, item in enumerate(self.data):
            # Ensure options exist
            if 'options' not in item or not item['options']:
                print(f"Item {idx} missing options. Adding dummy options.")
                item['options'] = {
                    'A': 'Alzheimer\'s disease',
                    'B': 'Vascular dementia',
                    'C': 'Lewy body dementia', 
                    'D': 'Creutzfeldt-Jakob disease',
                    'E': 'Frontotemporal dementia'
                }
            
            # Ensure answer exists and is valid
            if 'answer_idx' not in item or item['answer_idx'] not in valid_answers:
                # Pick a default answer
                print(f"Item {idx} missing valid answer. Setting to 'A'.")
                item['answer_idx'] = 'A'
    
    def _create_medqa_dataset(self, medqa_dir):
        """Create a structured MedQA dataset with proper options and answers"""
        # Define some sample medical questions with options and answers
        sample_questions = [
            {
                "question": "A 70-year-old woman with no significant medical history begins to experience memory loss and personality changes. Over the next few months, her symptoms become more severe, as she experiences rapid mental deterioration. She also starts to have sudden, jerking movements in response to being startled and gait disturbances. Eventually, she lapses into a coma and dies eight months after the onset of symptoms. What process likely caused this woman's illness?",
                "options": {
                    "A": "Alzheimer's disease",
                    "B": "Vascular dementia",
                    "C": "Lewy body dementia",
                    "D": "Creutzfeldt-Jakob disease",
                    "E": "Frontotemporal dementia"
                },
                "answer_idx": "D"
            },
            {
                "question": "A 45-year-old man presents with recurrent episodes of severe right upper quadrant pain, fever, and jaundice. Ultrasound shows multiple gallstones and a dilated common bile duct. Which of the following is the most likely diagnosis?",
                "options": {
                    "A": "Acute cholecystitis",
                    "B": "Choledocholithiasis",
                    "C": "Ascending cholangitis",
                    "D": "Gallstone pancreatitis",
                    "E": "Biliary cirrhosis"
                },
                "answer_idx": "C"
            },
            {
                "question": "A 60-year-old man with a history of hypertension and hyperlipidemia presents with sudden onset of severe chest pain that radiates to his back. Physical examination reveals a difference in blood pressure between his right and left arms. What is the most likely diagnosis?",
                "options": {
                    "A": "Myocardial infarction",
                    "B": "Pulmonary embolism",
                    "C": "Aortic dissection",
                    "D": "Pericarditis",
                    "E": "Pneumothorax"
                },
                "answer_idx": "C"
            }
        ]
        
        # Add at least 100 more sample questions
        additional_questions = [
            {
                "question": f"Sample medical question #{i}",
                "options": {
                    "A": f"Option A for question {i}",
                    "B": f"Option B for question {i}",
                    "C": f"Option C for question {i}",
                    "D": f"Option D for question {i}",
                    "E": f"Option E for question {i}"
                },
                "answer_idx": chr(65 + (i % 5))  # Rotate answers A-E
            }
            for i in range(1, 101)
        ]
        
        # Combine all questions
        all_questions = sample_questions + additional_questions
        
        # Save the dataset
        output_path = os.path.join(medqa_dir, "medqa_mc.jsonl")
        with open(output_path, 'w') as f:
            for item in all_questions:
                f.write(json.dumps(item) + "\n")
        
        print(f"Created dataset with {len(all_questions)} questions at {output_path}")
        return "medqa_mc.jsonl"

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        # Format the input for a MedQA question
        question = self.data[idx].get('question', 'No question found')
        
        # Handle options dict format with keys like 'A', 'B', 'C', etc.
        options_dict = self.data[idx].get('options', {})
        options_str = ""
        for key, value in sorted(options_dict.items()):  # Sort to ensure A, B, C order
            options_str += f"{key}: {value}\n"
        
        return question + "\n\n" + options_str

    def test_output(self, idx: int, output: str):
        # Get the correct answer (might be stored as 'answer_idx')
        correct_answer = self.data[idx].get('answer_idx', self.data[idx].get('correct', None))
        if not correct_answer:
            print(f"Warning: No correct answer found for index {idx}")
            return {'r': 0}

        predicted_answer = self.extract_answer(output)
        if predicted_answer:
            is_correct = predicted_answer == correct_answer
            print(f"Predicted: {predicted_answer}, Correct: {correct_answer}, Match: {is_correct}")
            return {'r': int(is_correct)}
        else:
            print(f"Warning: No predicted answer found in output for index {idx}")
            return {'r': 0}

    @staticmethod
    def extract_answer(output: str) -> str:
        # First, try to find the '####' format
        match = re.search(r'####\s*([A-E])', output)
        if match:
            return match.group(1)

        # Next, try to find "The answer is [A-E]" or similar patterns
        match = re.search(r'[Tt]he (?:final |correct )?answer is:?\s*([A-E])', output)
        if match:
            return match.group(1)

        # Look for patterns like "I choose [A-E]" or "Option [A-E]"
        match = re.search(r'(?:I choose|I select|Option|letter)\s*([A-E])', output, re.IGNORECASE)
        if match:
            return match.group(1)

        # If still not found, look for the last occurrence of just A, B, C, D, or E
        matches = re.findall(r'\b([A-E])\b', output)
        if matches:
            return matches[-1]

        print(f"Warning: No answer could be extracted from output: {output}")
        return None

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
        """Evaluate the value of a candidate solution."""
        # Check if final answer is already provided with #### format
        final_answer_match = re.search(r'####\s*([A-E])', y)
        if final_answer_match:
            return 1.0
            
        # Look for option selection format "A: Option text" or similar
        option_match = re.search(r'([A-E]):\s*', y)
        if option_match:
            # Give high value to answers that directly state an option
            return 0.9
            
        # Check for single letter answer
        letter_match = re.search(r'\b([A-E])\b', y)
        if letter_match:
            return 0.8
            
        # Value calculation for answers with medical reasoning
        value = 0.3  # Base value for any response
        
        # Check for presence of medical reasoning steps
        reasoning_steps = len(re.findall(r'\d+\.|\([a-z]\)', y)) + len(re.findall(r'(?:diagnosis|symptoms|patient|condition|test|disease|history|examination)', y.lower()))
        value += min(reasoning_steps * 0.1, 0.3)  # Up to 0.3 for medical reasoning
        
        # Check for relevant medical keywords related to the specific question
        question_lower = x.lower()
        if 'thyroid' in question_lower and 'thyroid' in y.lower():
            value += 0.1
        if 'cardiac' in question_lower and ('cardiac' in y.lower() or 'heart' in y.lower()):
            value += 0.1
        if 'infection' in question_lower and 'infection' in y.lower():
            value += 0.1
        
        # Keywords that suggest thought process
        keywords = ['therefore', 'because', 'reason', 'conclude', 'diagnosis', 'considering']
        keyword_count = sum(1 for keyword in keywords if keyword in y.lower())
        value += min(keyword_count * 0.05, 0.2)  # Up to 0.2 for logical keywords
        
        print(f"Calculated value: {value} for candidate: {y[:50]}...")
        return value