from src.core.config import Settings
from .llm_enum import LLMEnum
from .providers.openai_provider import OpenAIProvider

class LLMProviderFactory:
    def __init__(self, config: Settings):
        self.config = config
    
    def create(self, provider: LLMEnum):

        if provider == "openai":
            return OpenAIProvider(
                api_key=self.config.OPENAI_API_KEY,
                generation_model_id=self.config.OPENAI_MODEL_ID,
                input_pricing=self.config.OPENAI_INPUT_PRICING,
                output_pricing=self.config.OPENAI_OUTPUT_PRICING,
                api_url=self.config.OPENAI_API_BASE_URL,
                default_input_max_characters=self.config.OPENAI_MAX_INPUT_CHARS,
                default_generation_max_output_token=self.config.OPENAI_MAX_OUTPUT_TOKENS,
                defult_generation_temperature=self.config.OPENAI_TEMPERATURE
            )
        
        # ── future providers ──────────────────────────────────────────────────
        # You can enable Cohere provider here when needed
        # if provider == LLMEnum.COHERE.value:
        #     return CohereProvider()

        raise ValueError(f"Unsupported llm provider: {provider!r}") 