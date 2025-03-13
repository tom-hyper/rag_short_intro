from openai import OpenAI
from pinecone import Pinecone
#from exercise_3.retrieve_chunks import retrieve_chunks
# Here we're using dotenv (pip install python-dotenv) to manage environment vars
from dotenv import load_dotenv
import os

load_dotenv()

oa = OpenAI(api_key=os.getenv('OPENAI_API_KEY_NEW'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# In this function we will create the system message and include the relevant context
# TODO Find the system message instruction you came up with in Exercise 1
# TODO Paste it into the f""" string below,
# TODO The completed code will merge the instruction and context data

def embed_chunk(text):

    response = oa.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )

    print(response)

    embedding = response.data[0].embedding

    return embedding


# This will be used to retrieve chunks from Pinecone
# Default chunks to retrieve is 5
def retrieve_chunks(query, no_of_chunks=5):

    # The query is embedded before querying Pinecone
    embedding = embed_chunk(query)
    index_name = 'proofs'

    index = pc.Index(index_name)

    response = index.query(
        vector= embedding,
        top_k=no_of_chunks,
        include_metadata=True
    )

    retrieved_chunks = ""

    for match in response['matches']:

        retrieved_chunks += "________________________________\n"
        retrieved_chunks += "EXCERPT\n"
        retrieved_chunks += "-------\n"
        retrieved_chunks += match['metadata']['chunk'] + '\n'

    return retrieved_chunks

def inject_context_data(context):
    # Edit this system message
    system_message = f"""   
    You are a chatbot who's purpose is to answer questions about flat earth theories. 
    Use the following context as your ONLY source of evidence:    

    {context}
    
    """

    return system_message


def respond_to_question(question):
    # TODO in exercise 3 you wrote some code to retrieve your data
    # TODO complete the line below replacing x...
    # TODO you will need to call the code you worte last time in this line
    # You could etihter do this by implementing your code within this class
    # Or calling it some other way, the choice is yours
    context_data = retrieve_chunks(question)
    system_message = inject_context_data(context_data)

    # Call the OpenAI API with your systems message and question
    response = oa.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
        ]
    )

    # Parse the response to an answer and return it
    return response.choices[0].message.content


# Call the chatbot to respond to a question and get the answer based on the static injected context
# Example question: If the earth is flat, why do you only see the top half of a ship

answer = respond_to_question(
    'Has the Smithsonian Institute embraced Flat Earth theory?')

print('Answer:', answer)
