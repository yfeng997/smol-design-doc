import os
import json
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import tiktoken
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document




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
        verbose=True,
    )
    res = chain({"input_documents": documents})
    return res['output_text']
    

def summarize(filepath):
    with open(filepath, "r") as f:
        code = f.read()

    # HACK: 15000 roughly gives 3600 tokens
    if report_tokens(code) > 3600:
        code = code[:15000]

    llm = ChatOpenAI(temperature=0.5, model_name=GPT_3_5_TURBO)
    prompt = PromptTemplate(
        input_variables=["code"],
        template="""
Give one line summary of below with key functionality and components. Limit to 20 words.
{code}
""",
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)

    try:
        s = chain.run(code)
    except Exception as e:
        s = str(e)[-200:]

    return s.strip()


def traverse_summarize(root):
    summary = {}

    for path, dirs, files in os.walk(root):
        relpath = os.path.relpath(path, root)
        if relpath == ".":
            path_names = []
        else:
            path_names = relpath.split("/")

        curr = summary
        for p in path_names:
            next_curr = curr.get(p, {})
            curr[p] = next_curr
            curr = next_curr

        for f in files:
            if is_valid_file(f):
                curr[f] = summarize(os.path.join(path, f))

    return summary


def generate_design_doc(summary):
    """
    Given a summary of the codebase, generate a design doc.
    """
    llm = ChatOpenAI(temperature=0.5, model_name=GPT_3_5_TURBO_16k)
    prompt = PromptTemplate(
        input_variables=["summary"],
        template="""
Write a full technical design doc for the below encoded codebase. 

At a high level, discuss the purpose and functionalities of the codebase, 
major tech stack used, and an overview of the architecture. 
Describe the framework and languages used for each tech layer and corresponding 
communication protocols. If there's any design unique about this codebase, 
make sure to discuss those aspect in closer detail. For example, for a machine 
learning ops library, it's important to talk about compute orchestration, 
training and inference pipeline and model versioning.

Then in more details, describe the mission critical API endpoints. 
Describe the overall user experience and product flow. 
Talk about the data storage and retrieval strategy, including 
performance considerations and specific table schema. Touch on the deployment 
flow and infrastructure set up. Include topics around scalability, fault 
tolerance and monitoring.

Lastly, briefly touch on the security and authentication aspect. 
Talk about potential future improvements and enhancement to the feature set.

Codebase is encoded as follows:
- File name maps to a summary of the file content
- Folder name maps to files and subfolders in the folder

Encoded codebase is:
{summary}
""",
    )
    chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    try:
        return chain.run(summary=summary)
    except Exception as e:
        print(str(e))
        return ""


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


def report_tokens(s):
    encoding = tiktoken.encoding_for_model(GPT_3_5_TURBO)
    return len(encoding.encode(s))


def report_tokens_file(path):
    with open(path, "r") as f:
        s = f.read()
    return report_tokens(s)


def report_tokens_folder(path):
    total_tokens = 0
    if os.path.isdir(path):
        for p, dirs, files in os.walk(path):
            for f in files:
                if is_valid_file(f):
                    total_tokens += report_tokens_file(os.path.join(p, f))
    else:
        total_tokens += report_tokens_file(path)
    return total_tokens


if __name__ == "__main__":
    # d = traverse_summarize("/Users/yuansongfeng/Desktop/dev/civitai")
    # save_json(d, "summary.json")

    # print(report_tokens_folder("/Users/yuansongfeng/Desktop/dev/SimpML/summary.json"))

    # with open(
    #     "/Users/yuansongfeng/Desktop/dev/civitai/src/server/services/tag.service.ts",
    #     "r",
    # ) as f:
    #     s = f.read()
    #     print(report_tokens(s[:15000]))

    # d = load_json("part_summary.json")
    # design = generate_design_doc(d)
    # print(design)
    # save_txt(design, "design.txt")

    docs = documents_from_dir("/Users/yuansongfeng/Desktop/dev/civitai/src/libs")
    summary = summarize_documents(docs)
    save_txt(summary, "generated/civitai.txt")