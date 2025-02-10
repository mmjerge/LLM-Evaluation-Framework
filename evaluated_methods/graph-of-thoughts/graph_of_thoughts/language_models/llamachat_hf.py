# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Ales Kubicek

import os
import torch
from typing import List, Dict, Union
from .abstract_language_model import AbstractLanguageModel
import transformers

class Llama2HF(AbstractLanguageModel):
    def __init__(self, config_path: str = "", model_name: str = "llama7b-hf", cache: bool = False) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]
        self.model_id: str = self.config["model_id"]
        self.prompt_token_cost: float = self.config["prompt_token_cost"]
        self.response_token_cost: float = self.config["response_token_cost"]
        self.temperature: float = self.config["temperature"]
        self.top_k: int = self.config["top_k"]
        self.max_tokens: int = self.config["max_tokens"]

        os.environ["TRANSFORMERS_CACHE"] = self.config["cache_dir"]

        hf_model_id = self.model_id  # Use self.model_id directly

        model_config = transformers.AutoConfig.from_pretrained(hf_model_id)
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(hf_model_id)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.model.eval()
        torch.no_grad()

        self.generate_text = transformers.pipeline(
            model=self.model, tokenizer=self.tokenizer, task="text-generation"
        )

    def query(self, query: str, num_responses: int = 1) -> List[Dict]:
        if self.cache and query in self.respone_cache:
            return self.respone_cache[query]
        sequences = []
        query = f"<s><<SYS>>You are a helpful assistant. Always follow the instructions precisely and output the response exactly in the requested format.<</SYS>>\n\n[INST] {query} [/INST]"
        for _ in range(num_responses):
            sequences.extend(
                self.generate_text(
                    query,
                    do_sample=True,
                    top_k=self.top_k,
                    num_return_sequences=1,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=self.max_tokens,
                )
            )
        response = [
            {"generated_text": sequence["generated_text"][len(query) :].strip()}
            for sequence in sequences
        ]
        if self.cache:
            self.respone_cache[query] = response
        return response

    def get_response_texts(self, query_responses: List[Dict]) -> List[str]:
        return [query_response["generated_text"] for query_response in query_responses]