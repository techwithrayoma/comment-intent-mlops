from abc import ABC, abstractmethod

class LLMInterface(ABC):
    
    @abstractmethod
    def construct_prompt(self, prompt:str, role: str):
        pass

    @abstractmethod
    def estimate_cost(self, prompt_tokens:int, rocompletion_tokensle: int):
        pass

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int=None, 
                      temperature: float = None):
        pass
