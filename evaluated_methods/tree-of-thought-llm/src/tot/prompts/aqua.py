standard_prompt = """
Solve the following algebraic word problem:
{input}
Please output as a final letter answer.
"""

cot_prompt = """
Solve the following algebraic word problem step by step:
{input}
Please output as a final letter answer.
"""

propose_prompt = """
Here's an algebraic word problem:
{input}
Current solution steps:
{partial_solution}
What should be the next step in solving this algebraic problem?
"""

value_prompt = """
Here's an algebraic word problem:
{input}
Partial solution:
{partial_solution}
How likely is this partial solution to lead to the correct answer? (impossible/unlikely/likely/very likely/certain)
"""

value_last_step_prompt = """
Here's an algebraic word problem:
{input}
Proposed final answer:
{answer}
How likely is this answer to be correct? (impossible/unlikely/likely/very likely/certain)
"""