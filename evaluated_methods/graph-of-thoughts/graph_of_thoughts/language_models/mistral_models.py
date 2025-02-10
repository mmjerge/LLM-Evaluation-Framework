import backoff
import os
import random
import logging
logging.basicConfig(level=logging.DEBUG)
import time
from typing import List, Dict, Union
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from .abstract_language_model import AbstractLanguageModel

class Mistral(AbstractLanguageModel):
    def __init__(self, config_path: str = "", model_name: str = "open-mixtral-8x22b", cache: bool = False) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.max_tokens: int = self.config["max_tokens"]
        self.api_key: str = os.getenv("MISTRAL_API_KEY", self.config["api_key"])
        if self.api_key == "":
            raise ValueError("MISTRAL_API_KEY is not set")
        self.client = MistralClient(api_key=self.api_key)
    
    @backoff.on_exception(backoff.expo, Exception, max_tries=8)
    def query(self, query: str, num_responses: int = 1) -> Union[List[Dict], Dict]:
        logging.debug(f"Query input: {query}")
        logging.debug(f"Number of responses: {num_responses}")
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]

        responses = []
        for _ in range(num_responses):
            try:
                chat_response = self.client.chat(
                    model=self.model_id,
                    messages=[ChatMessage(role="user", content=query)],
                    max_tokens=self.max_tokens
                )
                responses.append({"content": chat_response.choices[0].message.content})
            except Exception as e:
                logging.warning(f"Error in Mistral API call: {e}")
                time.sleep(random.randint(1, 3))

        if self.cache:
            self.respone_cache[query] = responses if num_responses > 1 else responses[0]

        logging.debug(f"Query response: {responses}")
        return responses if num_responses > 1 else responses[0]
    
    @backoff.on_exception(backoff.expo, Exception, max_time=10, max_tries=6)
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
                api_messages = [ChatMessage(role=message["role"], content=message["content"]) for message in messages]

                chat_response = self.client.chat(
                    model=self.model_id,
                    messages=api_messages,
                    max_tokens=self.max_tokens
                )
                responses.append(chat_response)
                
                prompt_tokens = sum(len(m.content.split()) for m in api_messages)
                completion_tokens = len(chat_response.choices[0].message.content.split())
                
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens

            except Exception as e:
                logging.warning(f"Error in Mistral API call: {e}")

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
            if isinstance(response, dict) and 'choices' in response:
                content = response['choices']
                if isinstance(content, list) and len(content) > 0:
                    text = content[0].message.content
                    results.append(text)
                elif isinstance(content, str):
                    results.append(content)
            else:
                logging.warning(f"Unexpected response type: {type(response)}")
                results.append(str(response))
        
        logging.debug(f"get_response_texts output: {results}")
        return results