from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
import asyncio
import aiohttp
import csv
import math
import random
import time
from utils import process_zip_files_to_faiss  # Assuming this is a local utility function

async def fetch(session, url):
    async with session.get(url) as response:
        return await response.json()

async def download_file(session, url, output_path):
    async with session.get(url) as response:
        if response.status == 200:
            with open(output_path, 'wb') as f:
                f.write(await response.read())
            return "downloaded"
        else:
            return "error when downloading"

async def process_subquery(session, subquery, keyword, OUTPUT_FOLDER, repositories):
    URL = "https://api.github.com/search/repositories?q="
    QUERY = "q=" + keyword + "+is:featured"
    SUB_QUERIES = ["+created%3A<=2024-03-31", "+created%3A>=2022-12-01"]
    PARAMETERS = "&per_page=1"

    url = URL + QUERY + SUB_QUERIES[subquery - 1] + PARAMETERS
    data = await fetch(session, url)

    if data is None or 'total_count' not in data:
        print(f"Error: Unexpected API response for URL: {url}")
        return

    total_count = data['total_count']
    if total_count == 0:
        print(f"No repositories found for query {url}")
        return

    numberOfPages = min(int(math.ceil(total_count / 100.0)), 2)
    print(f"No. of pages = {numberOfPages}")

    download_results = []

    for currentPage in range(1, numberOfPages + 1):
        print(f"Processing page {currentPage} of {numberOfPages} ...")
        page_url = url + "&page=" + str(currentPage)
        page_data = await fetch(session, page_url)

        if page_data is None or 'items' not in page_data:
            print(f"Error: No valid items found in response for page {currentPage}")
            continue

        tasks = []
        for item in page_data['items']:
            user = item['owner']['login']
            repository = item['name']
            print(f"Downloading repository '{repository}' from user '{user}' ...")
            clone_url = item['clone_url']
            fileToDownload = clone_url.replace(".git", "/archive/refs/heads/master.zip")
            fileName = item['full_name'].replace("/", "#") + ".zip"
            output_path = OUTPUT_FOLDER + fileName

            tasks.append(download_file(session, fileToDownload, output_path))

            download_results.append([user, repository, clone_url, "downloaded"])  # Collect results

        await asyncio.gather(*tasks)  # Wait for all downloads to complete

    repositories.writerows(download_results)

async def make_db(result3, keywords):
    print("MAKING DB")
    OUTPUT_FOLDER = "/kaggle/working/"
    OUTPUT_CSV_FILE = "/kaggle/working/repositories.csv"  # Path to the CSV file generated as output

    csv_file = open(OUTPUT_CSV_FILE, 'w', newline='')  # Added newline='' for better CSV handling
    repositories = csv.writer(csv_file, delimiter=',')

    async with aiohttp.ClientSession() as session:
        for keyword in keywords:
            for subquery in range(1, len(["+created%3A<=2024-03-31", "+created%3A>=2022-01-01"]) + 1):
                await process_subquery(session, subquery, keyword, OUTPUT_FOLDER, repositories)

    csv_file.close()

    # Process zip files and create the database
    folder_path = '/kaggle/working/'  # Update with your actual path
    all_texts, file_paths = process_zip_files_to_faiss(folder_path)
    tavily_client = TavilyClient(api_key="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")
    taivily_docs = []
    prompt = PromptTemplate(
        template=(
            "You are doing some technical research from some web documents"
            "You are a Senior business developer at TAS Design Group Inc. , you have to summarize the key idea from some document for later use. You MUST extract key solutions and some effectiveness"
            "Transalte all information into english"
            "Follow the format exactly as instructed.\n"

            "{query}\n"
        ),
        input_variables=["query"],
    )

    chain = prompt | llm

    for keyword in keywords[:3]:
        # Perform various searches and collect URLs
        response = tavily_client.search(f"Some product or research about: {keyword}", search_depth="advanced", topic="general", max_results=1)
        for x in response['results']:
            taivily_docs.append(x['url'])

        # Remove duplicates by converting the list to a set and then back to a list
        web_path = list(set(taivily_docs))

        # Iterate over the web paths, load documents, and concatenate their content
        for x in web_path:
            print(x)
            loader = WebBaseLoader(
                web_paths=(x,)
            )
            docs = loader.load()

            # Combine all the document content into one string
            for doc in docs:
                start_time = time.time()  # {{ edit_1 }}
                if len(doc.page_content) >= 128000:
                    chunks = semantic_text_splitter.create_documents([doc.page_content])
                    for chunk in chunks:
                        if len(chunk.page_content) >= 128000:
                            chunks = text_splitter.create_documents([doc.page_content])
                            continue
                        chunk_result = chain.invoke(chunk.page_content)  # Removed timeout
                        all_texts.append(chunk_result.content)
                else:
                    doc_result = chain.invoke(doc.page_content)  # Removed timeout
                    all_texts.append(doc_result.content)
                if time.time() - start_time > 8:  # {{ edit_2 }}
                    continue  # Skip if processing takes more than 8 seconds
    prompt5 = PromptTemplate(
    template=(
            "As an Business Developer for TAS Design Group Inc, you're tasked with analyzing various repositories. Extract and summarize key information while considering human impact:"
            "1. Identify the core problem and its comprehensive solution, focusing on how it improves people's lives."
            "2. Highlight ethical technologies used in the solution, emphasizing user privacy and data security."
            "3. Evaluate the solution's accessibility and inclusivity for diverse user groups."
            "4. Assess the long-term societal and environmental impact of the solution."
            "5. Provide a concise, valuable summary for easy retrieval (NO SOURCE CODE)."
            "The information should be valuable for the task of business developement process"
            "{query}\n"
        ),
        input_variables=["query"],
    )
    chain5 = prompt5 | llm

    docs = []
    print(len(all_texts))
    random.shuffle(all_texts)
    count = 0
    for x in all_texts[:5]:
        print(count)
        start_time = time.time()  # Start timing
        if len(x) > 128000:
            chunks = semantic_text_splitter.create_documents(x)
            for chunk in chunks:
                if len(chunk.page_content) >= 128000:
                    chunks = text_splitter.create_documents(x)
                    continue
                result5 = chain5.invoke(chunk)
                docs.append(result5.content)
                if time.time() - start_time > 5:  # Check if processing time exceeds 5 seconds
                    print("Skipping due to timeout")
                    break
        else:
            result5 = chain5.invoke(x)
            docs.append(result5.content)
            if time.time() - start_time > 10:  # Check if processing time exceeds 10 seconds
                print("Skipping due to timeout")
                break
        count += 1
        if count == 15: break

    documents = []
    encoder = HuggingFaceEmbeddings()
    for x in docs:
        item = Document(page_content=x)
        documents.append(item)
    db = FAISS.from_documents(documents=documents, embedding=encoder)
    print("DB DONE")
    return db