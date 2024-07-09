
# Immune Deficiency Detection from Therapy Texts

This project aims to classify immune deficiency in children based on therapy texts extracted from medical documents (PDFs). It involves extracting therapy information, generating synthetic data, training a DistilBERT model, and implementing a Retrieval-Augmented Generation (RAG) system.

## Project Structure

```
immune_deficiency_detection/
│
├── data/
│   ├── 0001 - първична консултация.pdf
│   ├── 0002 - първична консултация.pdf
│   ├── 0003 - първична консултация.pdf
│   ├── 0001 - вторична консултация.pdf
│   ├── 0002 - вторична консултация.pdf
│   ├── 0003 - вторична консултация.pdf
│   └── therapy_classification.csv
│
├── src/
│   ├── __init__.py
│   ├── data_extraction.py
│   ├── synthetic_data.py
│   ├── create_csv.py
│   ├── train_model.py
│   ├── validate_model.py
│   ├── rag_implementation.py
│
├── notebooks/
│   ├── data_extraction.ipynb
│   ├── synthetic_data.ipynb
│   ├── create_csv.ipynb
│   ├── train_model.ipynb
│   ├── validate_model.ipynb
│   └── rag_implementation.ipynb
│
├── results/
│   └── model/
│       └── best_model/
│
├── README.md
└── requirements.txt
```

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/immune_deficiency_detection.git
    cd immune_deficiency_detection
    ```

2. **Create a virtual environment and activate it**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Data Extraction

Run the script to extract therapy information from PDFs:
```bash
python src/data_extraction.py
```

This will extract the therapy sections from the provided PDFs and save the extracted data in `data/extracted_therapies.json`.

## Generate Synthetic Data

Run the script to generate synthetic therapy data:
```bash
python src/synthetic_data.py
```

This will generate synthetic therapy records and save them in `data/synthetic_therapies.json`.

## Create CSV File

Combine the extracted and synthetic data into a CSV file:
```bash
python src/create_csv.py
```

This will create `data/therapy_classification.csv` with the necessary columns: `instruction`, `input`, and `output`.

## Train the Model

Train the DistilBERT model using the CSV file:
```bash
python src/train_model.py
```

This script will train the model and save the results in the `results/` directory.

## Validate the Model

Evaluate the trained model:
```bash
python src/validate_model.py
```

This script will output the accuracy, precision, recall, and F1 score of the model.

## Implement RAG

Set up and run the RAG system:
```bash
python src/rag_implementation.py
```

This script will index the documents in Elasticsearch, retrieve relevant documents, and use the trained model to answer queries.

## Project Notebooks

Jupyter notebooks are available for each step of the process in the `notebooks/` directory for interactive development and testing.

## Results

The best-trained model will be saved in the `results/model/best_model/` directory.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Elasticsearch](https://www.elastic.co/elasticsearch/)
- [Sentence-Transformers](https://www.sbert.net/)
