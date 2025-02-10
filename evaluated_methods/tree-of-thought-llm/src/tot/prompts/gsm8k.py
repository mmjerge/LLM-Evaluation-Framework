standard_prompt = """
Solve the following math word problem:
{input}
"""

cot_prompt = """
Solve the following math word problem step by step:
{input}
"""

propose_prompt = """
Here's a math word problem:
{input}

Current solution steps:
{partial_solution}

What should be the next step in solving this problem?
"""

value_prompt = """
Here's a math word problem:
{input}

Partial solution:
{partial_solution}

How likely is this partial solution to lead to the correct answer? (impossible/unlikely/likely/very likely/certain)
"""

value_last_step_prompt = """
Here's a math word problem:
{input}

Proposed final answer:
{answer}

How likely is this answer to be correct? (impossible/unlikely/likely/very likely/certain)
"""