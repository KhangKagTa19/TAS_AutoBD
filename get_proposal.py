from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain import PromptTemplate

def make_proposal(result3, db, company_name):

    class Feature(BaseModel):
        name: str = Field(description="Feature name")
        description: str = Field(description="Feature description")

    class Proposal(BaseModel):
        name: str = Field(description="Proposal product name")
        reason: str = Field(description="Why it is need for customer company")
        marketing_trends: str= Field(description="Why it is good in this marketing trends")
        stakeholder_mapping: str= Field(description=" Suitable stakeholders")
        key_features: List[Feature] = Field(description="All features that are suitable for the customer company")
    print("PREPARING EMAIL")
    pydantic_parser = PydanticOutputParser(pydantic_object=Proposal)
    format_instructions = pydantic_parser.get_format_instructions()

    proposal_prompt = PromptTemplate(
    template=(
            "You are a Business Developer tasked with extracting specific information from a company document. You MUST propose a final proposal for a customer company that is based on the query. "
            "This proposal MUST be formal and include: \n"
            "1. Proposal product name\n"
            "2. All features suitable for the customer company\n"
            "3. Detailed technical solution suitable for the customer company\n"
            """
            The proposal MUST also include
            Researching industry trends, company history, competitors, and prospects

            Stakeholder mapping

            Identifying growth opportunities

            Brainstorming strategies

            Experimenting

            Lead generation and qualification

            Prospecting

            Data analysis

            """
            "Follow the format exactly as instructed.\n"
            "{format_instructions}\n{query}"
        ),
        input_variables=["query"],
        partial_variables={"format_instructions": format_instructions},
    )
    retriever = db.as_retriever()
    custom_rag_prompt = PromptTemplate(
        template=(
            "You are an Senior solution architect tasked with analyzing company documents and providing relevant information. "
            "Based on the following context and question, provide a concise summary that includes:\n"
            "1. Key features of the product or service\n"
            "2. Technical details or solutions\n"
            "3. Any specific benefits for the customer company\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
            "Response:"
        ),
        input_variables=["context", "question"]
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    rag_response = rag_chain.invoke(str(result3))
    rag_response_dict = {"query": rag_response}
    final_chain = proposal_prompt | llm
    final = final_chain.invoke(rag_response_dict)
    print("FINALIZE EMAIL")
    # Define the email prompt template
    email_prompt = PromptTemplate(
        template=(
            "As an Business Developer for TAS Design Group Inc., craft a thoughtful, formal email proposal to our customer. Focus on how our solution can positively impact their business and society. Use the following HTML template and fill in the content:\n\n"
            "<html><body style='font-family: Arial, sans-serif; line-height: 1.6; color: #333;'>"
            "<p>Dear {customer_name},</p>"
            "<p>I hope this email finds you well. We are TAS Design Group Inc., a Japan and Vietnam based consulting and data science company, are excited to present a tailored solution that aligns with your company's values and goals.</p>"
            "<ol>"
            "<li><p>Introduce our proposed solution, explaining how it addresses the specific needs we've identified in your company. Highlight how this solution can potentially improve your operations, customer experience, and overall positive impact.</p></li>"
            "<li><p>Detail the key features of our solution, emphasizing:</p>"
            "<ul>"
            "<li>How it enhances user experience and accessibility</li>"
            "<li>Its commitment to data privacy and security</li>"
            "<li>Its potential for promoting sustainability or social responsibility</li>"
            "<li>How it aligns with and supports your company's mission and values</li>"
            "</ul></li>"
            "<li><p>Conclude with a brief summary of the mutual benefits of this partnership and express enthusiasm for potential collaboration.</p></li>"
            "</ol>"
            "<p>We look forward to the opportunity to discuss this proposal further and explore how we can work together to create positive change.</p>"
            "<p>Best regards,TAS Design Group Inc.</p>"
            "</body></html>"
            "\n\nEnsure checking grammar and the content is clear, formal, and organized into paragraphs. Fill in the HTML template with appropriate content based on the following query:\n\n"
            "{query}"
        ),
        input_variables=["customer_name", "query"],
    )

    # Assuming formal_text contains the query string and company_name contains the customer name
    email_chain = email_prompt | llm
    email = email_chain.invoke({
        "query": final.content,  # Assuming 'final' is a Pydantic object with a 'content' attribute
        "customer_name": company_name
    })

    return email.content