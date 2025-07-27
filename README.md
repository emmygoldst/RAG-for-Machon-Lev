# RAG-for-Machon-Lev
This repository contains the code for a Retrieval-Augmented Generation (RAG) system tailored for Machon Lev, built for document-based question answering and evaluation.

## Project Structure

- **`requirements.txt`**  
  Lists all required Python libraries, such as GPTQ, SentenceTransformers, BERTScore, and TQDM.
- **`jct_scraped_pages.json`**  
  JSON file containing the raw content fed into the RAG model.
- **`eval_questions.json`**  
  JSON file with questions and corresponding reference answers for evaluating model performance.
- **`process.py`**  
  Processes the JSON file and splits content into manageable chunks. In this project, chunks are of size 150.
- **`EmbeddingFaiss.py`**  
  Generates embeddings from the processed chunks using a SentenceTransformer model (`all-mpnet-base-v2`) and stores them in a FAISS index.
- **`LLM.py`**  
  Initializes the tokenizer and large language model. The default model is [`MaziyarPanahi/Mistral-7B-Instruct-v0.2-GPTQ`](https://huggingface.co/MaziyarPanahi/Mistral-7B-Instruct-v0.2-GPTQ), optimized for use with Google Colab.
- **`Generator.py`**  
  Constructs the prompt, retrieves relevant chunks, and generates answers using the LLM.
- **`config.py`**  
  Stores default parameters for model configuration and paths. Users may import or override these settings. Parameters include:
  ```python
  bits = 4
  group_size = 128
  desc_act = False
  llmodel = 'MaziyarPanahi/Mistral-7B-Instruct-v0.2-GPTQ'
  file_path = 'jct_scraped_pages.json'
  embedding_model = 'all-mpnet-base-v2'
  chunksize = 150
  ```
- **`Evaluation.py`**  
  Accepts a JSON file of evaluation questions and uses BERTScore to compare model responses against reference answers.
- **`main.py`**  
  The entry point of the project. Accepts an LLM and embedder instance and runs the model in either single-question or evaluation mode.
---

## Example Usage (Google Colab)
```python
# Clone the repository and install dependencies:
!git clone https://github.com/emmygoldst/RAG-for-Machon-Lev.git
!pip install -r /content/RAG-for-Machon-Lev/requirements.txt

# Move into the project directory
%cd RAG-for-Machon-Lev/

# Import modules and create components
from process import Preprocessor
from EmbeddingFaiss import Embedder
from LLM import LLM
from config import BITS, GROUP_SIZE, DESC_ACT, FILE_PATH, LLMODEL, EMBEDDING_MODEL, CHUNKSIZE

data = Preprocessor(FILE_PATH, CHUNKSIZE)
chunks = data.content_chunks()
llm = LLM(LLMODEL, BITS, GROUP_SIZE, DESC_ACT)
embedder = Embedder(EMBEDDING_MODEL, chunks)

# Run the main pipeline
from main import *
if __name__ == "__main__":
    main(embedder, llm)
```

---

### Sample Interaction
#### Initial output:
```
Choose mode: [1] Single question | [2] Evaluate file
```

#### Mode 2: Evaluate from file
```
Enter path to eval file (default: eval_questions.json): 
...
Question 2:
When does the academic year begin for the International Program?
  
Reference Answer:
The first semester begins after Sukkot.
analyzing input...
generating response...
  
Model Answer:
The academic year for the International Program begins after Sukkot each year.
...
BERTScore Evaluation:
Precision: 0.447, Recall: 0.722, F1: 0.582
```

#### Mode 1: Ask a question
```
Please enter your question (or type 'quit' to exit): How old is Machon Tal?
...
Answer:
Machon Tal was founded in September 2000.
Sources:
[1] Graduation for tal students computers, business and accounting - Jerusalem College of Technology  - https://www.jct.ac.il/en/news-and-updates/graduation-for-tal-students-computers-business-and-accounting/
[2] Prof Bodenheimer z"l - Jerusalem College of Technology  - https://www.jct.ac.il/en/news-and-updates/prof-bodenheimer-zl/
[3] Gallery - Jerusalem College of Technology  - https://www.jct.ac.il/en/tal-campus-project/gallery/
Time taken: 11.55 seconds
```
---

## Download RAG Data File (Optional)

To access the RAG input file from Google Drive:
```python
!gdown https://drive.google.com/uc?id=1WNkfWLy3hjF0RHWYTtVACUb4B2VePvBI
```
