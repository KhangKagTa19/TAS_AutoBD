import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel, Field
from typing import List
from langchain import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader  # Assuming this is necessary
from tavily import TavilyClient

def get_company_information(company_name, another_url):
    print("CRAWLING")
    tavily_client = TavilyClient(api_key="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")

    web_path = []
    queries = [
        f"{company_name} Japanese company",
        f"{company_name} information in Japan",
        f"Lack of {company_name} in Japan",
        f"Staff information in Japan or email of {company_name}",
        f"{company_name} Contact in Japan",
        f"Production of {company_name} in Japan",
        f"Future plan of {company_name}"
    ]

    # Perform various searches and collect URLs concurrently
    def fetch_urls(query):
        response = tavily_client.search(query, search_depth="advanced", topic="general", max_results=2)
        return [x['url'] for x in response['results']]

    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_urls, queries))

    # Flatten the list of results
    web_path = list(set(url for sublist in results for url in sublist))
    web_path.append(another_url)
    random.shuffle(web_path)  # Shuffle to randomize the order of URLs
    print("CRAWL DONE")

    class Company(BaseModel):
        name: str = Field(description="Extract company name")
        email: List[str] = Field(default_factory=list, description="Extract company email(s)")  # Changed to List
        general_information: str = Field(description="General information about the company")
        current_needs: str = Field(description="What the company needs right now")
        lacking_areas: str = Field(description="Areas where the company is lacking")
        sales_staff_info: str = Field(description="Sales staff information")
        contact_information: str = Field(description="Contact information from 'Contact us' page")
        software_production: str = Field(description="Information about software or IT production of the company")
        industry_information: str = Field(description="Information about the field or industry of the company")
        financial_highlights: str = Field(description="Financial highlights information of the company")
        summary: str = Field(description="Very short summary about this information")
        website: str = Field(default="", description="Company website URL")  # Added website field
        phone_number: str = Field(default="", description="Company phone number")  # Added phone number field

    pydantic_parser = PydanticOutputParser(pydantic_object=Company)
    format_instructions = pydantic_parser.get_format_instructions()
    print("GENERATING")
    # Create a PromptTemplate to structure the instructions
    prompt = PromptTemplate(
        template=(
                """
                Market research is essential to business development, providing the data organizations need to understand their target market, identify new opportunities, and make strategic decisions regarding product development, pricing and distribution channels. AI can significantly accelerate market research practices. One example is the use of NLP tools that automate the analysis of large amounts of unstructured text data such as customer survey responses, social media posts, online reviews, forum content and more. Using these systems, businesses can efficiently uncover actionable insights about their customers, including brand perception, preferences and pricing sensitivity, as well as broader market trends and industry growth opportunities.
                """
                "You are a Business Developer at TAS Design Group Inc. and website crawler tasked with extracting specific information about ONE potential customer {company_name} from web documents. Extract the following information from given data:"
                "Translate all information into English if necessary."
                "1. {company_name}"
                "2. Company email(s)"
                "3. General information about the company"
                "4. Current needs of the company"
                "5. Areas where the company is lacking"
                "6. Sales staff email"
                "7. Contact information from 'Contact us' page"
                "8. Information about software or IT production of the company"
                "9. Information about the field or industry of the company"
                "10. Financial highlights of the company"
                "11. A very short summary of all this information"
                "The extract information should be suitable for the business development process"
                "Follow the format exactly as instructed."
                "{format_instructions}"
                "{query}"
            ),
        input_variables=["query", "company_name"],
        partial_variables={"format_instructions": pydantic_parser.get_format_instructions()},
    )

    chain = prompt | llm 

    prompt2 = PromptTemplate(
        template=(
            """
            Market researching is essential to business development, providing the data organizations need to understand their target market, identify new opportunities, and make strategic decisions regarding product development, pricing and distribution channels. AI can significantly accelerate market research practices. One example is the use of NLP tools that automate the analysis of large amounts of unstructured text data such as customer survey responses, social media posts, online reviews, forum content and more. Using these systems, businesses can efficiently uncover actionable insights about their customers, including brand perception, preferences and pricing sensitivity, as well as broader market trends and industry growth opportunities.
            """
            "You are a Senior Business Developer at TAS Design Group Inc. , you have to extract characteristics from some information of a company. You MUST extract that one company name and then MUST divide the company_information into many smaller characteristics"
            "1. MUST Retrieve the company email"
            "Then"
            "2. Provide a list of all characteristics from those information in format: Charateristics_name: some information"
            "Follow the format exactly as instructed.\n"
            "The extract information should be suitable for the business development process"
            "{query}\n"
        ),
        input_variables=["query"],
    )

    chain2 = prompt2 | llm 
    # Initialize an empty string to store combined content
    combined_docs_content = ""

    def process_url(url):
        loader = WebBaseLoader(web_paths=(url,))
        docs = loader.load()
        results = []
        for doc in docs:
            results = []
            start_time = time.time()  # {{ edit_1 }}
            if len(doc.page_content) >= 128000:
                chunks = semantic_text_splitter.create_documents([doc.page_content])
                for chunk in chunks:
                    if len(chunk.page_content) >= 128000:
                        chunks = text_splitter.create_documents([doc.page_content])
                        continue
                    # {{ edit_2 }}: Pass a dictionary instead of a string
                    chunk_result = chain.invoke({"query": chunk.page_content, "company_name": company_name})  
                    results.append(chunk_result.content)
            else:
                # {{ edit_3 }}: Pass a dictionary instead of a string
                doc_result = chain.invoke({"query": doc.page_content, "company_name": company_name})  
                results.append(doc_result.content)
            if time.time() - start_time > 8:  # {{ edit_4 }}
                continue  # Skip if processing takes more than 8 seconds
        return "\n".join(results)

    # Use ThreadPoolExecutor for concurrent URL processing
    with ThreadPoolExecutor(max_workers=10) as executor:
        url_results = list(executor.map(process_url, web_path))
    print("DONE STEP 1")
    combined_docs_content = "\n".join(url_results)

    # Extract company characteristics
    result2 = chain2.invoke(combined_docs_content)

    # Regular expression to find email addresses
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    emails = list(set(re.findall(email_pattern, result2.content)))

    return result2, emails