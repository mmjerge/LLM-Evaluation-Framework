standard_prompt = """
Please answer the following multiple choice question:
{input}
Provide your answer as a single letter (a, b, c, or d) corresponding to the correct option.
"""

cot_prompt = """
Please answer the following multiple choice question step-by-step:
{input}
After your explanation, provide your final answer as a single letter (a, b, c, or d) corresponding to the correct option.
"""

propose_prompt = """
We're solving the following multiple choice question:
{input}

Current solution steps:
{partial_solution}

What should be the next step in solving this problem? If you have enough information to determine the final answer, provide it as a single letter (a, b, c, or d) corresponding to the correct option.
"""

value_prompt = """
We're evaluating a solution to this multiple choice question:
{input}

Proposed solution:
{partial_solution}

How likely is this solution to lead to the correct answer? (impossible/unlikely/likely/very likely/certain)
"""

value_last_step_prompt = """
We're evaluating the final answer to this multiple choice question:
{input}

Proposed final answer:
{answer}

How likely is this answer to be correct? (impossible/unlikely/likely/very likely/certain)
"""