import os
import time
from typing import List, Dict, Union, Any
from .abstract_language_model import AbstractLanguageModel
from vllm import LLM, SamplingParams

class Llama3VLLM(AbstractLanguageModel):
    def __init__(self, config_path: str = "", model_name: str = "llama3-instruct-vllm", cache: bool = False) -> None:
        """
        Initialize a Llama 3 model using vLLM for efficient inference
        
        Args:
            config_path: Path to the configuration file
            model_name: Name of the model configuration to use
            cache: Whether to cache responses
        """
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        self.temperature: float = self.config["temperature"]
        self.top_k: int = self.config["top_k"]
        self.max_tokens: int = self.config["max_tokens"]
        
        self.llm = LLM(
            model=self.model_id,
            trust_remote_code=True,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            tensor_parallel_size=1, 
            max_model_len=self.max_tokens,
        )
        
        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_k=self.top_k,
            max_tokens=self.max_tokens,
        )
        
        print(f"Initialized Llama3VLLM with model: {self.model_id}")
    
    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        """
        Generate responses for the given query using vLLM
        
        Args:
            query: The input text to generate responses for
            num_responses: Number of different responses to generate
            
        Returns:
            List of dictionaries containing the generated text
        """
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]
        
        formatted_query = f"<s><<SYS>>You are a helpful assistant. Always follow the instructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {query} [/INST]"
        
        if num_responses > 1:
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_k=self.top_k,
                max_tokens=self.max_tokens,
                n=num_responses 
            )
        else:
            sampling_params = self.sampling_params
        
        start_time = time.time()
        
        outputs = self.llm.generate([formatted_query], sampling_params)
        
        response = []
        for output in outputs:
            for gen_output in output.outputs:
                response.append({"generated_text": gen_output.text.strip()})
        
        inference_time = time.time() - start_time
        print(f"Generation completed in {inference_time:.2f} seconds")
        
        if self.cache:
            self.respone_cache[query] = response
            
        return response
    
    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        """
        Extract the generated text from the query responses
        
        Args:
            query_responses: List of dictionaries containing the generated text
            
        Returns:
            List of generated text strings
        """
        return [query_response["generated_text"] for query_response in query_responses]