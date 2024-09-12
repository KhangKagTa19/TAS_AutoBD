import asyncio
from langchain import PromptTemplate

async def get_hypothesis_idea(characteristics: str):  # Make this function async
        print("HYPO")
        prompt3 = PromptTemplate(
        template=(
                """
                As an Business Developer for IT company TAS Design Group Inc, your role is to analyze market trends and customer needs with a human-centric approach. Consider the following:

                1. Ethical implications: Ensure the proposed solution aligns with ethical standards and societal values.
                2. User experience: Prioritize solutions that enhance the quality of life for end-users.
                3. Sustainability: Consider the long-term environmental and social impact of the proposed solution.
                4. Inclusivity: Ensure the solution caters to diverse user groups, including those with special needs.
                5. Privacy and security: Prioritize user data protection and transparency in data usage.

                Based on the given extract characteristics:
                1. Identify the core human needs and pain points of the customer.
                2. Select and recommend suitable software/AI/ML/IT solution that our company can produce to address these needs ethically and effectively.
                3. Explain how this solution will positively impact the users' lives and contribute to societal well-being.

                {query}
                """
            ),
            input_variables=["query"],
        )

        chain3 = prompt3 | llm
        result3 = chain3.invoke(characteristics)  # Remove await if chain3.invoke is not async
        prompt4 = PromptTemplate(
            template=(
                "You are a company Business Developer of an IT startup company in Japan, which provides software or AI-powered application. Based on the following company characteristics, generate THREE simple and general keywords related to potential software solutions for this company. Each keyword should be in lowercase, with spaces replaced by underscores, suitable for searching on GitHub."
                "Company characteristics: {query}"
                "Format your response as a numbered list, with each keyword on a new line, like this:"
                "1. keyword_one"
                "2. keyword_two"
                "3. keyword_three"
            ),
            input_variables=["query"],
        )

        chain4 = prompt4 | llm
        result4 = chain4.invoke(result3.content)  # Remove await if chain4.invoke is not async
        print(result4.content)

        content = result4.content

        # Split the content by lines and process each line
        lines = content.split("\n")
        keywords = []
        for line in lines:
            # Remove leading/trailing whitespace and any numbering
            cleaned_line = line.strip().lstrip("0123456789. ")
            if cleaned_line and not '@' in cleaned_line:  # Ignore lines with email addresses
                keywords.append(cleaned_line.lower().replace(" ", "_"))

        # Ensure we have at least one keyword, even if it's just a default one
        if not keywords:
            keywords = ["software_solution"]

        print(keywords)
        return result3, keywords[:3]
