from string import Template

SYSTEM_RULES = Template(
    "\n".join([
        "You are an NLP data parser specialized in text classification.",
        "You will be provided with a YouTube comment and a Pydantic schema.",
        "Your task is to extract the comment's intent as JSON following the schema.",
        "Only extract the intent, do not add explanations or extra text.",
        "Keep the output strictly in JSON format.",
    ])
)

USER_RULES = Template(
    "\n".join([
        "## YouTube Comment:",
        "$comment",
        "",
        "## Pydantic Schema:",
        "predicted_intent: Literal['Question','Statement','Complaint','Praise','Suggestion']",
        "",
        "Return JSON only."
    ])
)