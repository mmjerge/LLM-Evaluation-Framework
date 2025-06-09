standard_prompt = """
Answer the following medical board exam question:
{input}
Please provide the letter corresponding to the correct answer (e.g., A, B, C, D, or E).
"""

cot_prompt = """
Answer the following medical board exam question step by step:
{input}
Please reason through each option carefully, considering the patient's symptoms, clinical presentation, test results, and relevant medical knowledge. After analyzing each option, select the letter corresponding to the correct answer.
"""

propose_prompt = """
Here's a medical board exam question:
{input}
Current reasoning:
{partial_solution}
What clinical knowledge, diagnostic reasoning, or additional considerations should be applied next to solve this medical question?
"""

value_prompt = """
Here's a medical board exam question:
{input}
Current clinical reasoning:
{partial_solution}
How likely is this reasoning to lead to the correct diagnosis or answer? (impossible/unlikely/likely/very likely/certain)
Provide your assessment based on medical best practices and clinical evidence.
"""

value_last_step_prompt = """
Here's a medical board exam question:
{input}
Proposed final answer:
{answer}
How likely is this answer to be correct based on the clinical information provided and medical knowledge? (impossible/unlikely/likely/very likely/certain)
"""