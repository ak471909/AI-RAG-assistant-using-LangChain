# AI-rag-assistant-using-LangChain
A QA Bot that Leverages the LangChain and LLM to Answer Questions from Loaded Document

It simulates a real-world scenario where a bot is built that will leverage LangChain and a large language model (LLM) to answer questions based on content from loaded PDF documents.

# Setting up a virtual environment
Create a virtual environment. Using a virtual environment allows you to manage dependencies for different projects separately, avoiding conflicts between package versions.

In the terminal of your Cloud IDE, ensure that you are in the path /home/project, then run the following commands to create a Python virtual environment.

```
pip install virtualenv
virtualenv my_env # create a virtual environment named my_env
source my_env/bin/activate # activate my_env
```

# Installing necessary libraries 
To ensure seamless execution of your scripts, and considering that certain functions within these scripts rely on external libraries, it's essential to install some prerequisite libraries before you begin. For this project, the key libraries you'll need are Gradio for creating user-friendly web interfaces and IBM-watsonx-AI for leveraging advanced LLM models from the IBM watsonx API.

```
# installing necessary packages in my_env
python3.11 -m pip install \
gradio==4.44.0 \
ibm-watsonx-ai==1.1.2  \
langchain==0.2.11 \
langchain-community==0.2.10 \
langchain-ibm==0.1.11 \
chromadb==0.4.24 \
pypdf==4.3.1 \
pydantic==2.9.1
```

- gradio allows you to build interactive web applications quickly, making your AI models accessible to users with ease.
- ibm-watsonx-ai for using LLMs from IBM watsonx.ai.
- langchain, langchain-ibm, langchain-community for using relevant features from Langchain.
- chromadb for using the chroma database as a vector database.
- pypdf is required for loading PDF documents.


## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI-rag-assistant-using-LangChain.git
cd AI-rag-assistant-using-LangChain
```

2. Create and activate virtual environment:
```bash
python -m venv my_env
source my_env/Scripts/activate  # Windows
# source my_env/bin/activate    # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file and add your OpenAI API key:
```bash
cp .env.example .env
# Edit .env and add your actual API key
```

5. Run the application:
```bash
python qabot.py
```

6. Open your browser to http://127.0.0.1:7860
