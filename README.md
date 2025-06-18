# BARTSumm: Abstractive Scientific Text Summarization

## Overview
This repository contains the implementation of **BARTSumm**, a transformer-based model for abstractive scientific text summarization, as described in the thesis:

> *"BARTSumm: Bidirectional Encoder & Autoregressive Decoder Transformer-based Scientific Text Abstractive Summarization"*  
> *A. S. M. Montashir Fahim, Department of Computer Science & Engineering, Rajshahi University of Engineering & Technology (RUET), Bangladesh*

The project leverages the **BART (Bidirectional and Auto-Regressive Transformer)** model, fine-tuned on the `ccdv/pubmed-summarization` dataset, to generate concise, coherent, and factually accurate summaries of complex scientific articles.  
It addresses challenges like handling technical jargon, preserving key findings, improving fluency, and achieving competitive performance:

- **ROUGE-1 F1:** 0.4092  
- **ROUGE-2 F1:** 0.1760  
- **ROUGE-L F1:** 0.2607  
- **BLEU-4:** 0.1208  
- **METEOR:** 0.2902  

The repository also includes resources for multilingual summarization, with a focus on **Bengali**, demonstrating adaptability to low-resource languages.

## Features
- **Abstractive Summarization:** Generates human-like summaries by paraphrasing and synthesizing content.  
- **Domain-Specific Fine-Tuning:** Optimized for scientific texts, particularly biomedical literature.  
- **Multilingual Support:** Extends to low-resource languages like Bengali.  
- **Comprehensive Evaluation:** ROUGE, BLEU, METEOR, BERTScore metrics + human evaluations.  
- **Open-Source Resources:** Pre-trained models, fine-tuning scripts, and datasets included.

## Repository Contents
- `BARTsumm.ipynb`: Jupyter notebook with data preprocessing, fine-tuning, training, and evaluation.
- `datasets/`: Scripts for `ccdv/pubmed-summarization` and Bengali dataset preprocessing.
- `models/`: Pre-trained BART model checkpoints and fine-tuned weights.
- `scripts/`: Utility scripts for tokenization, training, and inference.
- `results/`: Evaluation metrics and sample summaries.

## Requirements
Ensure you have the following dependencies:
```

python>=3.8
torch>=1.10.0
transformers>=4.20.0
datasets>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0

````
Install dependencies:
```bash
pip install -r requirements.txt
````

## Usage

### Clone the Repository

```bash
git clone https://github.com/montashirFahim/BARTSumm.git
cd BARTSumm
```

### Prepare the Dataset

* Download `ccdv/pubmed-summarization` via Hugging Face datasets library.
* For the Bengali dataset, see `datasets/` directory for preprocessing scripts.

### Run the Notebook

1. Open `BARTsumm.ipynb` in Jupyter Notebook or JupyterLab.
2. Follow sections to:

   * Load and preprocess the dataset.
   * Fine-tune the BART model.
   * Evaluate with ROUGE, BLEU, METEOR, and BERTScore.
   * Generate summaries for new articles.

### Example Command for Inference

```python
from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("./models/bart_finetuned")
tokenizer = BartTokenizer.from_pretrained("./models/bart_finetuned")

input_text = "Your scientific article text here..."
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)
```

## Results

The fine-tuned BART model achieved:

* **ROUGE-1 F1:** 0.4092
* **ROUGE-2 F1:** 0.1760
* **ROUGE-L F1:** 0.2607
* **BLEU-4:** 0.1208
* **METEOR:** 0.2902
* **BERTScore F1:** 0.8594

*Strong semantic alignment noted in biomedical texts; further optimization may help with long or highly technical documents.*

## Future Work

* **Graph Transformer Integration:** Combine SciBERT + Graph Transformer Networks for long-text summarization.
* **Bengali Summarization:** Expand the dataset and apply cross-lingual transfer with XLM-RoBERTa.
* **Real-World Deployment:** Low-latency inference + integration into academic/healthcare systems.
* **Advanced Techniques:** Explore controllable summarization, knowledge-augmented models, and multimodal inputs.

## Citation

If you use this code or dataset, please cite:

```
A. S. M. Montashir Fahim, "BARTSumm: Bidirectional Encoder & Autoregressive Decoder Transformer-based Scientific Text Abstractive Summarization," 
Bachelor of Science Thesis, Department of Computer Science & Engineering, Rajshahi University of Engineering & Technology, 2025.
```

## Acknowledgments

Supervised by **Prof. Dr. Boshir Ahmed**, Department of Computer Science & Engineering, RUET.
Resources from Kaggle and the Hugging Face ecosystem were utilized.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

For questions or contributions, please open an issue or contact the author at `montashirfahim25@gmail.com`.

