Knowledge Graph Grounding in RAG to Reduce LLM Hallucinations
This repository contains the implementation described in the paper "Knowledge Graph Grounding to Reduce LLM Hallucinations in Retrieval-Augmented Generation (RAG)". It presents two distinct system designs: a Heavyweight LLM-based pipeline inspired by KGGen and a Lightweight model using distilled, smaller models to evaluate how grounding with knowledge graphs (KGs) can mitigate hallucinations in large language models.

Paper Overview
The paper investigates hallucination in LLMs and proposes using structured Knowledge Graphs during the RAG process to ground model responses. It compares two approaches:

Heavyweight Model: Leverages multiple GPT-4o calls for each step in KG construction.

Lightweight Model: Minimizes LLM calls using custom-trained models to build and cluster KGs.

Evaluations are performed on the SQuAD dataset, using metrics like Exact Match, F1 Score, and Runtime to compare grounding methods.

Main Project Structure

```
.
├── main.py              # Forward pass over lightweight model
├── Evaluation.py        # Evaluation metrics + graph generation
├── models/              # All small model builds and training scripts
├── requirements.txt     # Package dependencies
├── README.md            
└── KGGen_paper.pdf      # Paper this project was based on
```

Setup

1. Clone the repository:

```
git clone https://github.com/your-username/kg-grounding.git
cd kg-grounding
```

2. Set up environment and install packages:

``` 
python -m venv venv
source venv/bin/activate 
pip install -r requirements.txt
```

Running

If you would like to view the Lightweight KG output of a model text, or from a .pdf/.pptx run

```python main.py```

If you would like to recreate the graphs already included in the folder, run

```python Evaluation.py```

Models used

EntityDetection: Trained on CoNLL-2003
GenerateLabel: Trained on the News Category Dataset
LLMJudge: Trained on UCI Product Classification Dataset

All models are stored and called from the models/ directory.

Evaluation Details
The system uses SQuAD for benchmarking with the following metrics:

Exact Match (EM): Measures how many predictions match the ground truth exactly.

F1 Score: Token-level overlap for partial credit.

Runtime: Tracks compute efficiency (notably, LLM API calls were run on Google’s servers, which are even faster than locally running a smaller NN on a laptop).

Example Training Output Format: 
```
{
  "question": "...",
  "ground_truths": ["John Doe"],
  "predictions": {
    "Lightweight": {
      "answer": "John Doe",
      "time_seconds": 5.35,
      "accuracy_em": 1,
      "f1_score": 1.0
    }
  }
}
```


Contact

For questions or collaboration, feel free to reach out:

Email: Eric.Stiefe8@gmail.com

Thanks for reading!
