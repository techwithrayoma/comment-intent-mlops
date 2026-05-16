from enum import Enum

class LLMEnum(Enum):
    
    OPENAI = "openai"

class OpenAIEnum(Enum):
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"