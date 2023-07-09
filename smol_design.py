import os
import json
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import tiktoken
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.callbacks.openai_info import get_openai_token_cost_for_model



"""
A smol design doc generator for any open source project
"""

# 4k tokens context
GPT_3_5_TURBO = "gpt-3.5-turbo"
GPT_3_5_TURBO_16k = "gpt-3.5-turbo-16k"
# 8k tokens context
GPT_4 = "gpt-4"


def documents_from_dir(dirpath):
    docs = []
    for path, dirs, files in os.walk(dirpath):
        for filename in files:
            if is_valid_file(filename):
                full_path = os.path.join(path, filename)
                with open(full_path, "r") as f:
                    d = f.read()
                # append path to beginning of doc
                relpath = os.path.relpath(path, dirpath)
                d = f"{relpath}\n" + d
                docs.append(Document(page_content=d))
    return docs

def summarize_documents(documents):
    """Map-reduce summarize documents to a design doc
    """
    llm = ChatOpenAI(temperature=0.5, model_name=GPT_3_5_TURBO)
   
    map_prompt = """Give a one line summary of below with key functionality and components:
    {text}"""
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])
    
    collapse_prompt = """Give a concise summary of below with key functionality and components:
    {text}"""
    collapse_prompt_template = PromptTemplate(template=collapse_prompt, input_variables=["text"])
    
    reduce_prompt = """Write a full technical design doc given below summaries over each file in the codebase. 
    Include architecture overview, design considerations and low level details.  
    {text}"""
    reduce_prompt_template = PromptTemplate(template=reduce_prompt, input_variables=["text"])

    chain = load_summarize_chain(
        llm, 
        chain_type="map_reduce", 
        map_prompt=map_prompt_template,
        collapse_prompt=collapse_prompt_template,
        combine_prompt=reduce_prompt_template,
        return_intermediate_steps=True,
        verbose=False,
    )
    res = chain({"input_documents": documents})
    return res['output_text']


def is_valid_file(filename):
    return (
        filename.endswith(".py")
        or filename.endswith(".ts")
        or filename.endswith(".js")
        or filename.endswith(".tsx")
    )


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


def report_tokens(s, model=GPT_3_5_TURBO):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s))


def estimate_cost_from_documents(documents, model=GPT_3_5_TURBO):
    total_tokens = sum(report_tokens(d.page_content) for d in documents)
    return get_openai_token_cost_for_model(model, total_tokens)

if __name__ == "__main__":
    docs = documents_from_dir("/Users/yuansongfeng/Desktop/dev/civitai/src/libs")
    estimated_cost = estimate_cost_from_documents(docs)
    print(f"Estimated cost: ${estimated_cost}")

    summary = summarize_documents(docs)
    save_txt(summary, "generated/civitai.txt")