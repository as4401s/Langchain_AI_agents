from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

information = """Andrew Yan-Tak Ng (Chinese: 吳恩達; born 1976) is a British-American computer scientist and technology entrepreneur focusing on machine learning and artificial intelligence (AI). Ng was a cofounder and head of Google Brain and was the former Chief Scientist at Baidu, building the company's Artificial Intelligence Group into a team of several thousand people.

Ng is an adjunct professor at Stanford University (formerly associate professor and Director of its Stanford AI Lab or SAIL). Ng has also worked in the field of online education, cofounding Coursera and DeepLearning.AI. He has spearheaded many efforts to "democratize deep learning" teaching over 8 million students through his online courses. Ng is renowned globally in computer science, recognized in Time magazine's 100 Most Influential People in 2012 and Fast Company's Most Creative People in 2014. His influence extends to being named in the Time100 AI Most Influential People in 2023.

In 2018, he launched and currently heads the AI Fund, initially a $175-million investment fund for backing artificial intelligence startups. He has founded Landing AI, which provides AI-powered SaaS products.

On April 11, 2024, Amazon announced the appointment of Ng to its board of directors.

Biography
Ng was born in the United Kingdom, in 1976 to Ronald Paul Ng, a hematologist, and Tisa Ho, an arts administrator, who were both immigrants from Hong Kong. He has at least one brother. In his youth, Ng lived in Hong Kong and Singapore. Ng attended and graduated from Raffles Institution. During his high school years, he demonstrated exceptional mathematical ability, winning a Silver Medal at the International Mathematical Olympiad.

In 1997, he earned his undergraduate degree with a triple major in computer science, statistics, and economics from Carnegie Mellon University in Pittsburgh, Pennsylvania. Between 1996 and 1998 he also conducted research on reinforcement learning, model selection, and feature selection at the AT&T Bell Labs.

In 1998, Ng earned his master's degree in Electrical Engineering and Computer Science from the Massachusetts Institute of Technology (MIT) in Cambridge, Massachusetts. At MIT, he built the first publicly available, automatically indexed web-search engine for research papers on the web. It was a precursor to CiteSeerX/ResearchIndex, but specialized in machine learning.

In 2002, he received his Doctor of Philosophy (Ph.D.) in Computer Science from the University of California, Berkeley, under the supervision of Michael I. Jordan. His thesis is titled "Shaping and policy search in reinforcement learning" and is well-cited to this day.

He started working as an assistant professor at Stanford University in 2002 and as an associate professor in 2009.

He currently lives in Los Altos Hills, California. In 2014, he married Carol E. Reiley. They have two children: a daughter born in 2019 and a son born in 2021. The MIT Technology Review named Ng and Reiley an "AI power couple".
"""

if __name__ == "__main__":
    print('Hello Langchain')

    summary_template = """
    Given the information below about a person, provide:
    1. A short summary
    2. Two interesting facts about them

    Information:
    {information}
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatOllama(model="llama3.2")

    # Create a runnable sequence
    chain = summary_prompt_template | llm | StrOutputParser()

    # Generate the response
    res = chain.invoke({"information": information})

    # Print the result
    print(res)
