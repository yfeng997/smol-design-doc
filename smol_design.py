import os
import json
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.callbacks import get_openai_callback
from constant import GPT_4, GPT_3_5_TURBO, GPT_3_5_TURBO_16k
from utils import (
    documents_from_dir,
    estimate_cost_path,
)

"""
A smol design doc generator for any open source project
"""


def summarize_documents(documents):
    """Map-reduce summarize documents to a design doc"""
    llm = ChatOpenAI(temperature=0, model_name=GPT_3_5_TURBO)
    reduce_llm = ChatOpenAI(temperature=0, model_name=GPT_4)

    map_prompt = """Give a one line summary of below with key functionality and components:
    {text}"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    collapse_prompt = """Give a concise summary of below with key functionality and components:
    {text}"""
    collapse_prompt_template = PromptTemplate(
        template=collapse_prompt, input_variables=["text"]
    )

    reduce_prompt = """Write a full technical design doc given below summaries over each file in the codebase. 
    Include architecture overview, design considerations and interesting details.  
    {text}"""
    reduce_prompt_template = PromptTemplate(
        template=reduce_prompt, input_variables=["text"]
    )

    chain = load_summarize_chain(
        llm,
        reduce_llm=reduce_llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        collapse_prompt=collapse_prompt_template,
        combine_prompt=reduce_prompt_template,
        return_intermediate_steps=True,
        verbose=True,
    )
    # track token and dollar usage
    with get_openai_callback() as usage_callback:
        res = chain({"input_documents": documents})
        print(usage_callback)

    return res["output_text"]


def save_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def load_json(file_path):
    with open(file_path, "r") as file:
        json_data = json.load(file)
    return json_data


def save_txt(data, filepath):
    with open(filepath, "w") as file:
        file.write(data)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    path = "/Users/yuansongfeng/Desktop/dev/civitai"
    docs = documents_from_dir(path)
    # docs = documents_from_repo("https://github.com/civitai/civitai")
    estimated_cost = estimate_cost_path(path)

    confirmation = input(f"Estimated cost is ${estimated_cost}. Continue? [y/n] ")
    if confirmation == "y":
        summary = summarize_documents(docs)
        save_txt(summary, "generated/civitai.txt")
    else:
        print("Aborted.")
