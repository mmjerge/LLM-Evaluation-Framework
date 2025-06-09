standard_prompt = """
Question: {input[question]}
Privacy policy clause: {input[clause]}
Determine if this privacy policy clause contains enough information to answer the question. 
Please respond with only "Relevant" or "Irrelevant".
"""

cot_prompt = """
Question: {input[question]}
Privacy policy clause: {input[clause]}

Let's analyze whether this privacy policy clause contains enough information to answer the question:

1. First, identify what specific information the question is asking for.
2. Next, examine the privacy policy clause to determine what information it provides.
3. Assess whether the clause directly addresses the question, partially addresses it, or does not address it at all.

Based on this analysis, determine if the clause is "Relevant" or "Irrelevant" for answering the question.
"""

propose_prompt = """
Question: {input[question]}
Privacy policy clause: {input[clause]}

Current analysis:
{partial_solution}

What additional aspects of the privacy policy clause should be examined to determine if it contains information relevant to the question?
"""

value_prompt = """
Question: {input[question]}
Privacy policy clause: {input[clause]}

Current analysis:
{partial_solution}

How likely is this analysis to lead to the correct determination of whether the clause is relevant to the question? (impossible/unlikely/likely/very likely/certain)
"""

value_last_step_prompt = """
Question: {input[question]}
Privacy policy clause: {input[clause]}

Final determination:
{answer}

How likely is this determination to be correct based on the relationship between the question and the information provided in the clause? (impossible/unlikely/likely/very likely/certain)
"""