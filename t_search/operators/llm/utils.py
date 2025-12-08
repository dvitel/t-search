''' Base module for LLM prompt-based operators. 
    Provides utilities to request the model

    Required environment variables:
        OPENAI_API_KEY - for OpenAI models
        GOOGLE_API_KEY - for Google Gemini models    
'''

import os
from typing import Type, TypeVar

ops_descriptions = {
    "add": "binary addition (add arg1 arg2)",
    "sub": "binary subtraction (sub minuend subtrahend)",
    "mul": "binary multiplication (mul arg1 arg2)",
    "div": "binary division (div dividend divisor) - divisor != 0",
    "pow": "binary power (pow base exponent)",
    "neg": "unary negation (neg arg)",
    "inv": "unary inversion (inv arg) - computes 1/args",
    "exp": "unary exponential function (exp arg)",
    "log": "unary natural logarithm (log arg) - arg > 0",
    "sin": "unary sine in radians (sin arg)",
    "cos": "unary cosine in radians (cos arg)",
}

def get_model_provider(model: str) -> str:
    if model.startswith("gpt-") or model.startswith("text-"):
        return "openai"
    elif model.startswith("gemini-"):
        return "google"
    else:
        raise ValueError(f"Unknown model provider for model: {model}")

T = TypeVar('T')

class LLMCaller:
    ''' Proxy to model '''
    def __init__(self, model: str):
        self.model = model
        self.provider = get_model_provider(model)
        self.openai_client = None
        self.google_client = None
    def _call_openai(self, prompt: str, response_schema: Type[T]) -> tuple[T, dict]: 
        import openai
        if self.openai_client is None:
            self.openai_client = openai.OpenAI(
                api_key = os.environ["OPENAI_API_KEY"],
                organization = os.environ.get("OPENAI_ORG_ID", None),
            )
        response = self.openai_client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            text_format = response_schema
        )

        res = response.output_parsed

        meta = response.usage

        usage = {
            "input_tokens": meta.input_tokens,
            "output_tokens": meta.output_tokens,
            "total_tokens": meta.total_tokens,
        }

        return res, usage

    def _call_google(self, prompt: str, response_schema: Type[T]) -> tuple[T, dict]: 
        from google import genai
        if self.google_client is None:
            self.google_client = genai.Client(
                api_key = os.environ["GOOGLE_API_KEY"]
            )
        response = self.google_client.models.generate_content(
            model = self.model,
            contents = prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": response_schema                
            }
        )
        res = response.parsed
        meta = response.usage_metadata
        usage = {
            "input_tokens": getattr(meta, "prompt_token_count", 0),
            "output_tokens": getattr(meta, "candidates_token_count", 0),
            "total_tokens": getattr(meta, "total_token_count", 0),
        }
        return res, usage
    
    def __call__(self, prompt: str, response_schema: Type[T]) -> tuple[T, dict]: 
        if self.provider == "openai":
            return self._call_openai(prompt, response_schema)
        elif self.provider == "google":
            return self._call_google(prompt, response_schema)
        else:
            raise ValueError(f"Unknown model provider: {self.provider}")
