import json
from base_service import BaseRepository
from generate_data_service import GenerateDataService
import prompt as pt

base_service = BaseRepository()
generate_data_service = GenerateDataService(base_service=base_service)


origin_prompt = input("Insert your origin prompt: ") 
origin_response = input("Insert response of prompt from LLM: ") 

selected_llm = input("Select LLM Model (Gpt / LLaMA or else): ")
task_type = input("Choose task type (Tiered contrastive reasoning / Multi-Metric contrastive reasoning): ")

def run():
    task_types = ""
    if "multi-metric" in task_type.lower():
        task_types = "multi-metric"
        handled_data = generate_data_service.get_multi_metric_df(origin_prompt=origin_prompt,
                                                origin_response=origin_response)

        sample_prompt_list = []
        sample_response_list = []
        sample_score_list = []

        for cont in ["helpfulness","correctness","coherence","complexity","verbosity"]:
            if handled_data[f"{cont}_retrieved_prompts"].values[0] in sample_prompt_list:
                sample_response_list.append(handled_data[f"{cont}_retrieved_response"].values[0])
                sample_score_list[sample_prompt_list.index(handled_data[f"{cont}_retrieved_prompts"].values[0])].append(f"{cont}_score: {handled_data[f"{cont}_score"].values[0]}")
            else:
                sample_prompt_list.append(handled_data[f"{cont}_retrieved_prompts"].values[0])
                sample_response_list.append(handled_data[f"{cont}_retrieved_response"].values[0])
                sample_score_list.append([f"{cont}_score: {handled_data[f"{cont}_score"].values[0]}"])
        multi_metric = """

        """
        for p in range(len(sample_prompt_list)):
            multi_metric += f"Sample prompt {p+1}: {sample_prompt_list[p]}\n"
            multi_metric += f"Sample Score {p+1}: {sample_score_list[p]}"
        multi_metric_new_prompt = pt.multi_metric_prompt.format(origin_prompt=origin_prompt, origin_response=origin_response,multi_metric=multi_metric)
        response = base_service.llm_request(user_input=multi_metric_new_prompt,
                                            selected_llm=selected_llm)
        print("mm prompt: ",multi_metric_new_prompt)
        print("response: ",response)
        return multi_metric_new_prompt, response
    else:
        task_types = "Tiered"
        handled_data = generate_data_service.get_tiered_df(origin_prompt=origin_prompt,
                                                origin_response=origin_response)
        sample_data_prompt = pt.tiered_data_prompt.format(tiered_handled_data=handled_data.iloc[0])
        tiered_prompt = pt.tiered_prompt.format(origin_prompt=origin_prompt,
                                                origin_response=origin_response,
                                                sample_data=sample_data_prompt)
        response = base_service.llm_request(user_input=tiered_prompt,
                                            selected_llm=selected_llm)
        print("tiered prompt: ",tiered_prompt)
        print("response: ",response)
        return tiered_prompt, response, task_types
    
if __name__ == "__main__":
    prompt, response, task_types = run()
    with open("result.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"task_type":task_types,"final_response": response}, ensure_ascii=False) + "\n")