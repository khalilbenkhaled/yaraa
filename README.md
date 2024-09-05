**YARAA** (Yet Another Rag Automation Attempt) is a library that simplifies the development of RAG (Retrieval-Augmented Generation) pipelines. You can build your vector database in any way you prefer, and YARAA will assist you in testing, evaluating, and optimizing the best configurations.

![tutorial (2)](https://github.com/user-attachments/assets/34213228-969e-4627-a8e0-51858ebf599b)

# Installation

To install the library, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/khalilbenkhaled/yaraa
   ```

2. **Navigate to the project directory:**

   ```bash
   cd yaraa
   ```

3. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. **Install the library:**

   ```bash
   pip install .
   ```

   **Optional**: To install specific optional libraries, use:

   ```bash
   pip install .[NAME_OF_LIBRARY1, NAME_OF_LIBRARY2]
   #example:
   pip install .[ollama,chromadb,streamlit,fastapi]
   ```

   Supported options include:

   - `transformers`
   - `ollama`
   - `server` (for the OpenAI library)
   - `chromadb`
   - `streamlit`
   - `fastapi`

   Choose the options you need and include them in the command above.


## Usage

### Quick Start (for Fast Testing or New Users)

1. **Configure Your Settings:**

   - Locate the file named `example1.yaml` in the repository.
   - Edit this file to specify your hyperparameters, LLM, the query encoder, and the path to your vector database directory. 

   *Note:* Currently, the library offers limited configuration options, but more features (like multiple vector stores and additional hyperparameter controls) will be added soon.

2. **Run the Inference:**

   Execute the following command:

   ```bash
   python inference.py example1.yaml
   ```

   This command will launch a browser window with a chat interface for interacting with your RAG pipeline.

3. **File Information:**

   - **`example1.yaml`**: Controls inference settings.
   - **`example2.yaml`**: Shows how to use the YAML file for evaluating your RAG pipeline.

   For inference, the library currently supports:
   - Streamlit
   - FastAPI

   For evaluation, the only supported metric is:
   - Answer relevance by RAGAS.

### Advanced Usage (for Experienced Developers and Large Projects)

- For detailed usage and full control, explore the `examples` folder in the repository. It provides various examples to help you understand how to utilize the library in different scenarios.
