import os
import logging
from typing import Dict, List, Union
import backoff
from together import Together
from .abstract_language_model import AbstractLanguageModel

class ATogetherAI(AbstractLanguageModel):
    def __init__(self, config_path: str = "", model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", cache: bool = False) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.max_tokens: int = self.config["max_tokens"]
        self.api_key: str = os.getenv("TOGETHER_API_KEY", self.config.get("api_key", ""))
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY is not set")
        self.client = Together(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    def query(
            self, query: str, num_responses: int = 1
        ) -> Union[List[Dict], Dict]:
        """
        Query the Together AI model for responses.

        :param query: The query to be posed to the language model.
        :type query: str
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: Response(s) from the Together AI model.
        :rtype: Union[List[Dict], Dict]
        """
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]

        response = self.chat([{"role": "user", "content": query}], num_responses)

        if self.cache:
            self.respone_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, Exception, max_time=10, max_tries=6)
    def chat(self, messages: List[Dict], num_responses: int = 1) -> Dict:
        """
        Send chat messages to the Together AI model and retrieves the model's response.
        Implements backoff on exceptions.

        :param messages: A list of message dictionaries for the chat.
        :type messages: List[Dict]
        :param num_responses: Number of desired responses, default is 1.
        :type num_responses: int
        :return: The Together AI model's response.
        :rtype: Dict
        """
        try:
            # Set temperature to 0.7 if num_responses > 1, else default to 0
            temperature = 0.7 if num_responses > 1 else 0

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=self.max_tokens,
                n=num_responses,
                temperature=temperature
            )

            self.logger.info(f"Response from Together AI: {response}")
            return response.model_dump()
        except Exception as e:
            self.logger.error(f"Error in Together AI API call: {str(e)}")
            raise

    def get_response_texts(
        self, query_response: Union[List[Dict], Dict]
    ) -> List[str]:
        """
        Extract the response texts from the query response.

        :param query_response: The response dictionary (or list of dictionaries) from the Together AI model.
        :type query_response: Union[List[Dict], Dict]
        :return: List of response strings.
        :rtype: List[str]
        """
        if not isinstance(query_response, list):
            query_response = [query_response]
        return [
            choice['message']['content']
            for response in query_response
            for choice in response.get('choices', [])
        ]
