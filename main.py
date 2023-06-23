import streamlit
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
"""
To run, 
Create virtual environment 
install the required libraries
use streamlit run main.py
"""

class LLM:
    def __init__(self, model_path):
        """
        Initialize the LLM object.

        @param model_path (str): The path to the model.
        """
        self.model_path = model_path

    def load_model(self):
        """
        Load the language model.

        @return (object): The loaded language model.
        """
        self.model = GPT4All(model=self.model_path, verbose=True)
        return self.model

    def generate_prompt(self):
        """
        Generate the prompt template.

        @return (object): The prompt template object.
        """
        prompt = PromptTemplate(input_variables=['question'], template="""
            Question?: {question}
            
            Answer: Let's think step by step.
            """)
        return prompt

    def predict_results(self, prompt, model):
        """
        Predict results based on the given prompt and model.

        @param prompt (object): The prompt template object.
        @param model (object): The loaded language model.

        @return (str): The generated response.
        """
        llm_chain = LLMChain(prompt=prompt, llm=model)

        streamlit.title('🌟🔗 UNCENSORED LLM GPT')
        streamlit.info('This is using the MPT model!')
        prompt = streamlit.text_input('Enter your prompt here!')

        if prompt: 
            response = llm_chain.run(prompt)
            print(response)
            streamlit.write(response)


if __name__ == '__main__':
    PATH = '/Users/boadisamson/Library/Application Support/nomic.ai/GPT4All/ggml-mpt-7b-chat.bin'
    llm = LLM(model_path=PATH)
    loaded_model = llm.load_model()
    prompt_template = llm.generate_prompt()
    results = llm.predict_results(prompt=prompt_template, model=loaded_model)
