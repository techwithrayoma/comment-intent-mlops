from openai import OpenAI
from typing import Dict, List, Optional, Union

from ..llm_interface import LLMInterface
from ..llm_enum import OpenAIEnum


class OpenAIProvider(LLMInterface):
    def __init__(self, 
        api_key: str,
        generation_model_id: str,
        input_pricing: float,
        output_pricing: float,
        api_url: str=None, 
        default_input_max_characters: int=1000, 
        default_generation_max_output_token: int=20,
        defult_generation_temperature: float=0,
    ):
        
        self.api_key = api_key
        self.api_url = api_url

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_token = default_generation_max_output_token
        self.defult_generation_temperature = defult_generation_temperature

        self.generation_model_id = generation_model_id
        self.input_pricing = input_pricing
        self.output_pricing = output_pricing

        self.client = OpenAI(
            api_key=self.api_key, 
        )

        self.enums = OpenAIEnum


    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return (
            prompt_tokens * self.input_pricing +
            completion_tokens * self.output_pricing
        )
    
    def construct_prompt(self, prompt: str, role: str) -> Dict[str, str]:
        return {"role": role, "content": prompt[:self.default_input_max_characters].strip()}
    

    def generate_text(
        self,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Union[str, int]]:
        
        messages = chat_history or []

        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=messages,
            temperature=self.defult_generation_temperature,
            max_tokens=self.default_generation_max_output_token
        )

        msg_content = response.choices[0].message.content
        usage = response.usage

        return {
            "text": msg_content,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens
        }
