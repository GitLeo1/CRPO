import re
from rank_bm25 import BM25Okapi
import numpy as np
import openai
from openai import OpenAI
import replicate
import json
from dotenv import load_dotenv
load_dotenv()

class BaseRepository:
    def __init__(self):
        self.client = OpenAI()

    def llm_gpt4o(self,prompt="", user_input='', response_format=None):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ]
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0,
            response_format=response_format
        )
        return response.choices[0].message.content

    def llm_llama(self,prompt="", user_input=""):
        full_prompt = f"{prompt}\n{user_input}" if prompt else user_input
        out = replicate.run(
            "meta/meta-llama-3-8b-instruct",
            input={
                "prompt": full_prompt,
                "max_tokens": 512,
                "temperature": 0.0,
            },
        )
        if isinstance(out, str):
            return out
        if isinstance(out, list):
            return "".join([x for x in out if isinstance(x, str)])
        return str(out)

    def llm_gpt_o4(self,
        user_input: str = "",
    ) -> str:
        kwargs = {
            "model": "o4-mini",
            "messages": [{
            "role": "user",
            "content": [{"type": "text", "text": user_input}]
        }],
        }

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content

    def read_jsonl(self,jsonl_path):
        rows = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    
    def refiner_prompt_opt(self,refiner_prompt, origin_prompt,analyzed_result):
        refiner_prompt = refiner_prompt.format(prompt = origin_prompt,
                                               analyzed_result = analyzed_result)
        return refiner_prompt
    
    def llm_request(self,user_input,selected_llm):
        if "gpt" in selected_llm.lower():
            gpt_response = self.llm_gpt4o(user_input=user_input)
            return gpt_response
        else:
            llama_response = self.llm_llama(user_input=user_input)
            return llama_response
    
    def tokenize(self, text):
        text = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
        return [t for t in text.lower().split() if t]
    
    def ret_bm25(self, query, docs, top_k = 10):
        corpus_tokens = [self.tokenize(text=d) for d in docs]
        bm25 = BM25Okapi(corpus_tokens)
        q_tokens = self.tokenize(text=query)
        scores = bm25.get_scores(q_tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        top_k_docs = []
        top_k_scores = []
        for idx, s in ranked:
            top_k_docs.append(docs[idx])
            top_k_scores.append(f"{s:.3f}")
        return top_k_docs, top_k_scores