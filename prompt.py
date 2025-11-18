multi_metric_prompt = """
                    **Example Response Format:**
                {{"prompt": "[new_generated prompt]", "response":"[general response of new generated prompt]"}}

                There are NEVER need other explanation word. Just follow jsonl reponse format.
                You MUST Follow JSONL output format. Never need other explanation word.
                You just write one line of jsonl output. **Only One line of JSONL! NEVER OTHER!!!**

                Your task is to complement the existing prompts and responses given and write them with improved content. Your newly created prompts and responses are evaluated in five areas.
                1. Helpfulness: How useful and helpful the overall response is.
                2. Correctness: The response is based on facts, no hallucinations, no mistakes. The response covers everything required in the instruction.
                3. Coherence: The response is self-consistent in terms of content, style of writing, and does not contradict itself. The response can be logically followed and understood by a human. The response does not contain redundant or repeated information.
                4. Complexity: Rate the response along a simple to complex spectrum. A simple response uses simple, easy to understand vocabulary and sentence structure that children can understand. Conversely, a complex response uses sophisticated language with enhanced vocabulary that adults with advanced education or experts on the topic would use.
                5. Verbosity: A low verbosity response is direct to the point without extra wordings. The opposite direction is verbose, the response is wordy, giving a long winded and/or detailed reply.

                [The origin Prompt]
                {origin_prompt}

                [The origin response]
                {origin_response}

                You should refine this prompt and response.
                Here are some well-structured prompt sample and its scores. 
                These samples include prompt, and its specific score corresponding to them. 
                Identify the characteristics of the prompts, responses, and scores provided to create new prompts and examples of responses.
                {multi_metric}

                These are examples of good prompts and responses with high scores for each score item. Referring to the features of this prompt, reconstruct the original prompt and response to get a high score.
                """

tiered_prompt = """

                **Example Response Format:**
                {{"prompt": "[new_generated prompt]", "response":"[general response of new generated prompt]"}}

                There are NEVER need other explanation word. Just follow jsonl reponse format.
                You MUST Follow JSONL output format. Never need other explanation word.
                You just write one line of jsonl output. **Only One line of JSONL! NEVER OTHER!!!**

                Your task is to complement the existing prompts and responses given and write them with improved content. Your newly created prompts and responses are evaluated in five areas.
                1. Helpfulness: How useful and helpful the overall response is.
                2. Correctness: The response is based on facts, no hallucinations, no mistakes. The response covers everything required in the instruction.
                3. Coherence: The response is self-consistent in terms of content, style of writing, and does not contradict itself. The response can be logically followed and understood by a human. The response does not contain redundant or repeated information.
                4. Complexity: Rate the response along a simple to complex spectrum. A simple response uses simple, easy to understand vocabulary and sentence structure that children can understand. Conversely, a complex response uses sophisticated language with enhanced vocabulary that adults with advanced education or experts on the topic would use.
                5. Verbosity: A low verbosity response is direct to the point without extra wordings. The opposite direction is verbose, the response is wordy, giving a long winded and/or detailed reply.

                [The origin Prompt]
                {origin_prompt}

                [The origin response]
                {origin_response}

                You should refine this prompt and response.
                Here are some other prompt sample. These samples include prompt, response, and its specific score corresponding to them. Identify the characteristics of the prompts, responses, and scores provided to create new prompts and examples of responses.

                Definitions:
                - MAX: the sample with the highest mean score across the five metrics (best case).
                - MEDIAN: the sample whose mean score is closest to the group’s median (typical/representative case).
                - MIN: the sample with the lowest mean score (worst case).

                Your tasks:
                1) Identify what makes MAX strong, MEDIAN typical, and MIN weak.
                2) Propose a new, improved prompt and a plausible response that would score higher than MEDIAN and approach MAX.
                3) Keep outputs concise and grounded in the observed patterns.

                {sample_data}
"""

tiered_data_prompt = """
                This is example for well-scored. Catch the strength point from given prompt and response.
                Good prompt: {tiered_handled_data.max_retrieved_prompt}
                Response of Good prompt: {tiered_handled_data.max_retrieved_response}
                score: [helpfulness: {tiered_handled_data.max_helpfulness}, correctness: {tiered_handled_data.max_correctness}, coherence: {tiered_handled_data.max_coherence}, complexity: {tiered_handled_data.max_complexity}, verbosity: {tiered_handled_data.max_verbosity}]
                ----
                This is example for not bad but not good. Figure out the balance from given sample.
                median prompt: {tiered_handled_data.median_retrieved_prompt}
                Response of Good prompt: {tiered_handled_data.median_retrieved_response}
                score: [helpfulness: {tiered_handled_data.median_helpfulness}, correctness: {tiered_handled_data.median_correctness}, coherence: {tiered_handled_data.median_coherence}, complexity: {tiered_handled_data.median_complexity}, verbosity: {tiered_handled_data.median_verbosity}]
                ----
                This is example for weakness. Figure out and avoid the weakness from the sample.
                Bad prompt: {tiered_handled_data.min_retrieved_prompt}
                Response of bad prompt: {tiered_handled_data.min_retrieved_response}
                score: [helpfulness: {tiered_handled_data.min_helpfulness}, correctness: {tiered_handled_data.min_correctness}, coherence: {tiered_handled_data.min_coherence}, complexity: {tiered_handled_data.min_complexity}, verbosity: {tiered_handled_data.min_verbosity}]
"""