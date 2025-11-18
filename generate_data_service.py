import pandas as pd
import numpy as np
import json
from base_service import BaseRepository


class GenerateDataService:
    def __init__(self, 
                 base_service:BaseRepository):
        self.base_service = base_service

    def _get_train_data(self):
        train_rows = []
        with open("helpsteer_train.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    train_rows.append(json.loads(line))
        return train_rows

    def _rag_service(self, origin_prompt):
        train_rows = self._get_train_data()
        train_prompt_list = [cont["prompt"] for cont in train_rows]
        rag_result = self.base_service.ret_bm25(query=origin_prompt, docs=train_prompt_list, top_k=10)
        return rag_result
    
    def rag_result_df(self, rag_result, train_rows, origin_prompt, origin_response):
        bm25_result_df = pd.DataFrame({"valid_prompt":[],
                                    "valid_response":[],
                                    "retrieved_prompt":[],
                                    "retrieved_response":[],
                                    "helpfulness":[],
                                    "correctness":[],
                                    "coherence":[],
                                    "complexity":[],
                                    "verbosity":[]
                                    })
        train_prompt_list = [cont["prompt"] for cont in train_rows]
        for idx in range(len(rag_result[0])):
                score_list = list(train_rows[train_prompt_list.index(rag_result[0][idx])].values())[2:]
                row_ = {
                        "valid_prompt": f"{origin_prompt}",
                        "valid_response":f"{origin_response}",
                        "retrieved_prompt": f"{rag_result[0][idx]}",
                        "retrieved_response":f"{rag_result[0][idx]}",
                        "helpfulness": score_list[0],
                        "correctness": score_list[1],
                        "coherence": score_list[2],
                        "complexity": score_list[3],
                        "verbosity": score_list[4],
                        }
                bm25_result_df.loc[len(bm25_result_df)] = row_
        return bm25_result_df

    def make_tiered_data(self, df, k):
        SCORE_COLS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
        K = k

        for c in SCORE_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.copy()
        df["set_id"] = np.arange(len(df)) // K
        df["mean_score"] = df[SCORE_COLS].mean(axis=1)

        def summarize_one_set(g: pd.DataFrame) -> pd.Series:
            m = g["mean_score"]
            idx_max = m.idxmax()
            idx_min = m.idxmin()
            idx_med = (m - m.median()).abs().idxmin()
            out = {
                "max_row_idx": int(idx_max),
                "max_mean": float(df.at[idx_max, "mean_score"]),
                "max_retrieved_prompt": df.at[idx_max, "retrieved_prompt"],
                "max_retrieved_response":df.at[idx_max, "retrieved_response"],
                "median_row_idx": int(idx_med),
                "median_mean": float(df.at[idx_med, "mean_score"]),
                "median_retrieved_prompt": df.at[idx_med, "retrieved_prompt"],
                "median_retrieved_response":df.at[idx_med, "retrieved_response"],
                "min_row_idx": int(idx_min),
                "min_mean": float(df.at[idx_min, "mean_score"]),
                "min_retrieved_prompt": df.at[idx_min, "retrieved_prompt"],
                "min_retrieved_response":df.at[idx_min, "retrieved_response"],
            }
            for c in SCORE_COLS:
                out[f"max_{c}"] = float(df.at[idx_max, c]) if pd.notna(df.at[idx_max, c]) else np.nan
                out[f"median_{c}"] = float(df.at[idx_med, c]) if pd.notna(df.at[idx_med, c]) else np.nan
                out[f"min_{c}"] = float(df.at[idx_min, c]) if pd.notna(df.at[idx_min, c]) else np.nan
            return pd.Series(out)

        tiered_df = (
            df.groupby("set_id", as_index=False)
            .apply(summarize_one_set)
            .reset_index(drop=True))
        return tiered_df
    
    def make_multi_metric_data(self, df,k):
        SCORE_COLS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
        df["set_id"] = np.arange(len(df)) // k
        def pick_max_all(g: pd.DataFrame) -> pd.Series:
            out = {"set_id": int(g["set_id"].iloc[0])}
            for m in SCORE_COLS:
                mmax = g[m].max()
                top = g.loc[g[m] == mmax, ["retrieved_prompt", "retrieved_response"]]
                out[f"{m}_score"] = mmax
                out[f"{m}_retrieved_prompts"] = top["retrieved_prompt"].tolist()[0]
                out[f"{m}_retrieved_response"] = top["retrieved_response"].tolist()[0]
            return pd.Series(out)

        mm_df = (
            df.groupby("set_id", as_index=False)
            .apply(pick_max_all)
            .reset_index(drop=True)
        )
        return mm_df
    
    def get_tiered_df(self, origin_prompt, origin_response):
        rag_result = self._rag_service(origin_prompt=origin_prompt)
        train_rows = self._get_train_data()
        df = self.rag_result_df(rag_result=rag_result, 
                                train_rows=train_rows, 
                                origin_prompt=origin_prompt, 
                                origin_response=origin_response)
        return self.make_tiered_data(df, k=10)

    def get_multi_metric_df(self,origin_prompt, origin_response):
        rag_result = self._rag_service(origin_prompt=origin_prompt)
        train_rows = self._get_train_data()
        df = self.rag_result_df(rag_result=rag_result, 
                                train_rows=train_rows, 
                                origin_prompt=origin_prompt, 
                                origin_response=origin_response)
        return self.make_multi_metric_data(df, k=10)