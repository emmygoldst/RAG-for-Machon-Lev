# RAG-for-Machon-Lev
This repository contains the code for a Retrieval-Augmented Generation (RAG) system designed for Machon Lev.

## Project Description
- requirements.txt: A file containing all the needed libraries for the models, such as GPTQ, sentence transformers, bert score and TQDM.
- jct_scraped_pages.json: This is a JSON file containing the input for the RAG model.
- eval_questions.json: This is a JSON file containing questions and answers for the model to compare and get accuracy scores from.
- process.py: This is a python code to process a given JSON format file and split it into chunks of a chosen size. In our project we chose 150.
- EmbeddingFaiss.py: This is a python code that uses the chunks from the proceess codes and creates embeddings with a selected model and creates a Faiss index from those embeddings. We chose 'all-mpnet-base-v2' as an embedding model.
- LLM.py: This is a python code used to initialize the tokenizer and large language model to use to generate responses. We chose a modified Mistral-7B from hugging face to fit our google colab: 'MaziyarPanahi/Mistral-7B-Instruct-v0.2-GPTQ'.
- Generator.py: This is the python code to generate the response. It first builds a prompt, extracts the sources of the selected chunks, and runs the pipeline connecting the prompt with the LLM.
- config.py: This is a python file containing the default parameters for the models' initializations. The user must import the file to use its contents but also the user can set up his own parameters, which is why it is set in a different file. The parameters include bit size, group size, act order (False for faster results, True for more accurate results), and the model name for the llm and tokenizer. In addition, it includes the file path for the jct scraped pages, which is the RAG file, the chunk size for the preprocessing steps, and the embedding model name.
~~~
default settings:
  bits=4
  group_size=128
  desc_act=False
  llmodel='MaziyarPanahi/Mistral-7B-Instruct-v0.2-GPTQ'
  file_path = 'jct_scraped_pages.json'
  embedding_model = 'all-mpnet-base-v2'
  chunksize=150
~~~
- Evaluation.py: This is a python code that accpets a JSON file with evaluation questions and answers to compare the model answers to. It uses BERT score for the comparison and scores.The user can input his own evaluation file, or use the default file as mentioned above.
- main.py: The main function that runs the whole model. It gets two values: the predefined llm and the predefined embedder model with embeddings and all. 

## Example usage in google colab:
```python
# clone and install requirements:
!git clone https://github.com/emmygoldst/RAG-for-Machon-Lev.git
!pip install -r /content/RAG-for-Machon-Lev/requirements.txt

# set up the embedding model and llm and preprocess the file based on the default values:
%cd RAG-for-Machon-Lev/
from process import Preprocessor
from EmbeddingFaiss import Embedder
from LLM import LLM
from config import BITS, GROUP_SIZE, DESC_ACT, FILE_PATH, LLMODEL, EMBEDDING_MODEL, CHUNKSIZE
data = Preprocessor(FILE_PATH, CHUNKSIZE)
chunks = data.content_chunks()
llm = LLM(LLMODEL, BITS, GROUP_SIZE, DESC_ACT)
embedder = Embedder(EMBEDDING_MODEL, chunks)

# Run the main code:
from main import *
if __name__ == "__main__":
    main(embedder, llm)
'''
The first output line is this:
  Choose mode: [1] Single question | [2] Evaluate file
Example output for 2:
  Enter path to eval file (default: eval_questions.json): 
  Question 2:
  When does the academic year begin for the International Program?
  
  Reference Answer:
  The first semester begins after Sukkot.
  analyzing input...
  generating response...
  
  Model Answer:
  The academic year for the International Program begins after Sukkot each year.
  
  Retrieved Context Chunks:
  
  Chunk 1:
  Calendar The International Program follows the Jerusalem College of Technologys standard academic calendar which can be viewed by clicking here. The college provides breaks for the Jewish holidays. The first semester of the program starts after Sukkot each year. Registration To register for the coming academic year please complete the simple online registration form . Please note there is a 300 shekel registration fee which is non-refundable. Contact Us Coordinator Men's Program: Mr. Gavriel Novick Email: gnovickjct.ac.il Phone: 972-58-419-0087 American Line: 929-242-1119 Coordinator Women's Program Mrs. Bracha Berger Email: bbergerjct.ac.il Phone: 972-58-627-4087 International Program Office Email: ESPjct.ac.il Phone: 972-2-675-1011 Opening of the program is contingent on the number of registered students. The program is subject to changes, at the discretion of the Jerusalem College of Technology.
  (Source: Programs in English - Jerusalem College of Technology )
  ----------------------------------------
  
  Chunk 2:
  The International program in English is proud to present a novel opportunity to obtain a prestigious academic degree which provides strong professional training in the areas of business and computers. Requirements for Acceptance High School Diploma SAT ACT or TIL (Israeli) Exam Personal Interview For computer science, a strong background in Math (pre-calculuscalculus) Tuition Approximately 4,000 per year Depending on the exchange rate and current public tuition rates set by the Council for Higher Education. This fee does not include room, board, and Beit Midrash, if required. Academic Calendar Dates The International Program follows the Jerusalem College of Technologys standard academic calendar which can be viewed by clicking here . The college provides breaks for the Jewish holidays. The first semester of the program starts after Sukkot each year. Registration To register for the coming academic year please complete the simple online registration form .
  (Source: International Program in English - Jerusalem College of Technology )
  ----------------------------------------
  
  Chunk 3:
  JCT opens the 2019 school year on campus Home Page News and Updates ... JCT opens 2019 school year on campus JCT kicked off the academic year on September 1st with festive events at the colleges various campuses. 06.09.19 JCT kicked off the academic year on September 1 st with festive events at the colleges various campuses. Lev Campus held a special gathering in the Beit Midrash. Following the main ceremony, students continued to learn in chevrutot (pairs and small groups) to prepare for the auspicious month of Elul, together with their rabbis and teachers. The Tvuna program for Haredim- a recognized leader of academic programs for ultra-Orthodox women- opened its academic year with a significant increase in the number of students and a welcome ceremony with Rabbinate Miriam Weinberg.
  (Source: JCT opens 2019 school year on campus - Jerusalem College of Technology )
  ----------------------------------------
  
  Chunk 4:
  When? February Duration: 4 Years Place: קמפוס לב Nursing Nursing Holders of a nursing degree are involved in improving the quality of life of patients throughout the life cycle. Nurses also provide solutions to the problems of special... When? Duration: Pharmacy Pharmacy The pharmacy program at the Jerusalem College of Technology works in collaboration with the pharmacy degree program at Hebrew University, and at the end students receive... When? Elul Semester Duration: Four Years Business Management Accounting and Information Systems Accounting and Information Systems What is Accounting and Information Systems? The studies in the Accounting and Information Systems Department prepare the graduates to fulfill key positions in accounting,... When? Fall Semester Duration: 3 years fourth year supplementary for accountant license (for those interested).
  (Source: Academic Programs - Jerusalem College of Technology )
  ----------------------------------------
  Some weights of RobertaModel were not initialized from the model checkpoint at roberta-large and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
  You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
  Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
  
  BERTScore Evaluation:
  Precision: 0.447, Recall: 0.722, F1: 0.582

Example output for 1:
  Please enter your question (or type 'quit' to exit): How old is Machon Tal?
  Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
  Processing your question...
  analyzing input...
  generating response...
  Retrieved chunks:
  Chunk 1:
  130 graduates from Machon Tal Home Page News and Updates ... Graduation for tal students computers, business and accounting This week, the Lev Academic Center held a graduation ceremony for 130 graduates in computer science, business administration, and accounting from Machon Tal. April 2024 This week, the Jerusalem College of Techonology held a graduation ceremony for 130 graduates in computer science, business administration, and accounting from Machon Tal. Assaf Yazdi, Director-General of the Jerusalem Affairs and Heritage Ministry who spoke as the guest of honor at the event, noted the contribution of JCT to the development of Jerusalem, given its uniqueness as an academic institution which combines religious studies with high-level academic studies. Among the graduating students were also the sisters Avital Kortzman (Galinski), who currently works at Moovit, and Reut Galinski, who won the Rector's Prize.
  (Source: Graduation for tal students computers, business and accounting - Jerusalem College of Technology )
  --------------------------------------------------
  Chunk 2:
  Joseph Bodenheimer was the driving force behind Machon Levs dynamic growth as the only college in Israel combining the academic disciplines of sophisticated technology, contemporary business and management, with intensive Jewish studies. Under his leadership, Machon Naveh for Haredi men and Machon Lustig for Haredi women were established. He went on to found Machon Tal in September 2000, the first academic program to offer religious women the opportunity to study engineering, health care and management in a religious environment and created special programs for new immigrants from Ethiopia, Russia, France, South America and English-speaking countries. In 2019, Prof. Bodenheimer was named a Yakir Yerushalayim with the awarding of the Jerusalem Citizenship Award by Mayor Moshe Lion and the Jerusalem Municipality. The Jerusalem College of Technology mourns the loss of a charismatic and brilliant leader. May his memory will serve as a blessing to us always.
  (Source: Prof Bodenheimer z"l - Jerusalem College of Technology )
  --------------------------------------------------
  Chunk 3:
  Gallery Renditions of the new Tal Campus. Home Page Tal Campus Project ...
  (Source: Gallery - Jerusalem College of Technology )
  --------------------------------------------------
  Answer:
  Machon Tal was founded in September 2000.
  Sources:
  [1] Graduation for tal students computers, business and accounting - Jerusalem College of Technology  - https://www.jct.ac.il/en/news-and-updates/graduation-for-tal-students-computers-business-and-accounting/
  [2] Prof Bodenheimer z"l - Jerusalem College of Technology  - https://www.jct.ac.il/en/news-and-updates/prof-bodenheimer-zl/
  [3] Gallery - Jerusalem College of Technology  - https://www.jct.ac.il/en/tal-campus-project/gallery/
  Time taken: 11.55 seconds
  ------------------------------------------------------------
  Please enter your question (or type 'quit' to exit): quit
  Exiting program.
'''
```

To access the RAG json file from google drive use this command:
``` python
!gdown https://drive.google.com/uc?id=1WNkfWLy3hjF0RHWYTtVACUb4B2VePvBI
```


