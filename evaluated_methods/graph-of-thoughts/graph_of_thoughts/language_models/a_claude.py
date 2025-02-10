import backoff
import os
import random
import logging
logging.basicConfig(level=logging.DEBUG)
import time
from typing import List, Dict, Union
from anthropic import Anthropic, AnthropicError
from .abstract_language_model import AbstractLanguageModel

class AClaude(AbstractLanguageModel):
    def __init__(self, config_path: str = "", model_name: str = "claude-3.5", cache: bool = False) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.max_tokens: int = self.config["max_tokens"]
        self.api_key: str = os.getenv("ANTHROPIC_API_KEY", self.config["api_key"])
        if self.api_key == "":
            raise ValueError("ANTHROPIC_API_KEY is not set")
        self.client = Anthropic(api_key=self.api_key)
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=8)
    def query(self, query: str, num_responses: int = 1) -> Union[List[Dict], Dict]:
        logging.debug(f"Query input: {query}")
        logging.debug(f"Number of responses: {num_responses}")
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]

        responses = []
        for _ in range(num_responses):
            try:
                message = self.client.messages.create(
                    model=self.model_id,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": query,
                        }
                    ],
                )
                responses.append({"content": message.content})
            except Exception as e:
                self.logger.warning(f"Error in Claude API call: {e}")
                time.sleep(random.randint(1, 3))

        if self.cache:
            self.respone_cache[query] = responses if num_responses > 1 else responses[0]

        logging.debug(f"Query response: {responses}")
        return responses if num_responses > 1 else responses[0]
    
    @backoff.on_exception(backoff.expo, AnthropicError, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> Union[Dict, List[Dict]]:
        logging.debug(f"Chat input messages: {messages}")
        logging.debug(f"Number of responses: {num_responses}")
        responses = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        api_params = {
            "model": self.model_id,
            "max_tokens": self.max_tokens,
        }

        for _ in range(num_responses):
            try:
                api_params["messages"] = messages

                response = self.client.messages.create(**api_params)
                responses.append(response)
                
                prompt_tokens = sum(len(m['content'].split()) for m in messages)
                completion_tokens = len(response.content.split())
                
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

            except AnthropicError as e:
                self.logger.warning(f"Error in Claude API call: {e}")

        self.prompt_tokens += total_prompt_tokens
        self.completion_tokens += total_completion_tokens
        prompt_tokens_k = float(self.prompt_tokens) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )

        logging.debug(f"Chat response: {responses}")
        return responses if num_responses > 1 else responses[0]

    def get_response_texts(self, query_response: Union[List[Dict], Dict]) -> List[str]:
        logging.debug(f"get_response_texts input: {query_response}")
        
        if not isinstance(query_response, list):
            query_response = [query_response]
        
        results = []
        for response in query_response:
            if isinstance(response, dict) and 'content' in response:
                content = response['content']
                if isinstance(content, list) and len(content) > 0:
                    # Assuming the first item contains the text we want
                    text = content[0].text if hasattr(content[0], 'text') else str(content[0])
                    results.append(text)
                elif isinstance(content, str):
                    results.append(content)
            else:
                logging.warning(f"Unexpected response type: {type(response)}")
                results.append(str(response))
        
        logging.debug(f"get_response_texts output: {results}")
        return results
    