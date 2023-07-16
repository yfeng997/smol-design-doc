import argparse
from constant import GPT_3_5_TURBO_16k, GPT_3_5_TURBO
import tiktoken
import os
from langchain.docstore.document import Document
from github import Github
from langchain.callbacks.openai_info import get_openai_token_cost_for_model


def count_token_string(s, model=GPT_3_5_TURBO):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(s, allowed_special={"<|endoftext|>"}))


def count_token_path(path, model=GPT_3_5_TURBO):
    if "github.com/" in path:
        docs = documents_from_repo(path)
    else:
        docs = documents_from_dir(path)
    return sum(count_token_string(d.page_content, model) for d in docs)


def estimate_cost_path(path, model=GPT_3_5_TURBO):
    tokens = count_token_path(path, model)
    return get_openai_token_cost_for_model(model, tokens)


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
                relpath = os.path.join(relpath, filename)
                docs.append(Document(page_content=d, metadata={"path": relpath}))
    return docs


def documents_from_repo(repo_url):
    if "github.com/" not in repo_url:
        raise Exception(f"Invalid repo url: {repo_url}")

    url = repo_url.split("github.com/")[1]
    access_token = os.environ["GITHUB_TOKEN"]
    g = Github(access_token)
    repo = g.get_repo(url)

    # Get all files in the codebase
    files = get_repo_files(repo)
    docs = []
    for file in files:
        content = read_file_contents(file)
        docs.append(Document(page_content=content, metadata={"path": file.path}))
    return docs


def get_repo_files(repo):
    files = []
    contents = repo.get_contents("")
    while contents:
        file_content = contents.pop(0)
        if file_content.type == "dir":
            contents.extend(repo.get_contents(file_content.path))
        else:
            if is_valid_file(file_content.path):
                files.append(file_content)
    return files


def read_file_contents(file):
    try:
        response = file.decoded_content
        return response.decode("utf-8")
    except:
        return ""


def is_valid_file(filename):
    valid_format = [".py", ".ts", ".js", ".tsx", ".mts", ".sh"]
    return any(filename.endswith(ext) for ext in valid_format)


def main():
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Smol Design CLI")
    parser.add_argument(
        "action",
        choices=["token", "cost"],
    )
    parser.add_argument("path", type=str, help="Local directory or repo URL")
    parser.add_argument("--model", type=str, help="Model name", default=GPT_3_5_TURBO)

    args = parser.parse_args()

    if args.action == "token":
        result = count_token_path(args.path, args.model)
    elif args.action == "cost":
        result = estimate_cost_path(args.path, args.model)

    print("Result:", result)


if __name__ == "__main__":
    main()
