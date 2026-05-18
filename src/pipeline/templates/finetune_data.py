from string import Template

SYSTEM_FINETUNE = Template(
    "You are a strict text classification system.\n"
    "Classify the comment into exactly one of these labels:\n"
    "Question, Complaint, Statement, Praise, Suggestion.\n"
    "Return ONLY valid JSON in this exact format:\n"
    "{\"predicted_intent\": \"<label>\"}\n"
    "Do not output anything else."
)

INSTRUCTION_FINETUNE = Template("Comment:\n$comment")