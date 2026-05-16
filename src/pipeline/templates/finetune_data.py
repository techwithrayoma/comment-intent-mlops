from string import Template

SYSTEM_FINETUNE = Template(
    "\n".join([
        "You are an NLP data parser specialized in text classification.",
        "Your task is to extract the intent of a comment.",
        "Only output JSON with the predicted intent.",
        "Do not add explanations.",
        "You MUST choose one of the following labels ONLY:",
        "Question, Complaint, Statement, Praise, Suggestion.",
        "Do not invent new labels."
    ])
)

INSTRUCTION_FINETUNE = Template(
    "## YouTube Comment:\n$comment\n\n## Extracted JSON:\n```json"
)

