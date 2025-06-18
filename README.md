BARTSumm: Abstractive Scientific Text Summarization
Overview
This repository contains the implementation of BARTSumm, a transformer-based model for abstractive scientific text summarization, as described in the thesis "BARTSumm: Bidirectional Encoder & Autoregressive Decoder Transformer-based Scientific Text Abstractive Summarization" by A. S. M. Montashir Fahim, submitted to the Department of Computer Science & Engineering, Rajshahi University of Engineering & Technology (RUET), Bangladesh.
The project leverages the BART (Bidirectional and Auto-Regressive Transformer) model, fine-tuned on the ccdv/pubmed-summarization dataset, to generate concise, coherent, and factually accurate summaries of complex scientific articles. The model addresses challenges such as handling technical jargon, preserving key findings, and improving fluency, achieving competitive performance with ROUGE-1 F1 of 0.4092, ROUGE-2 F1 of 0.1760, ROUGE-L F1 of 0.2607, BLEU-4 of 0.1208, and METEOR of 0.2902.
Additionally, the repository includes resources for multilingual summarization, with a focus on Bengali, a low-resource language, demonstrating the model's adaptability to diverse linguistic contexts.
Features

Abstractive Summarization: Generates human-like summaries by paraphrasing and synthesizing content, rather than extracting sentences.
Domain-Specific Fine-Tuning: Optimized for scientific texts, particularly biomedical literature, using the PubMed dataset.
Multilingual Support: Extends to low-resource languages like Bengali with a custom dataset.
Comprehensive Evaluation: Uses ROUGE, BLEU, METEOR, and BERTScore metrics, supplemented by human evaluations for semantic fidelity.
Open-Source Resources: Includes pre-trained models, fine-tuning scripts, and datasets for reproducibility.

Repository Contents

BARTsumm.ipynb: Jupyter notebook containing the complete implementation, including data preprocessing, model fine-tuning, training, and evaluation.
datasets/: Directory with scripts to load and preprocess the ccdv/pubmed-summarization dataset and the custom Bengali dataset.
models/: Pre-trained BART model checkpoints and fine-tuned weights.
scripts/: Utility scripts for tokenization, training, and inference.
results/: Output files with evaluation metrics and sample summaries.

Requirements
To run the code, ensure you have the following dependencies installed:
python>=3.8
torch>=1.10.0
transformers>=4.20.0
datasets>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0

Install dependencies using:
pip install -r requirements.txt

Usage

Clone the Repository:
git clone https://github.com/<your-username>/BARTSumm.git
cd BARTSumm


Prepare the Dataset:

The ccdv/pubmed-summarization dataset can be downloaded via the Hugging Face datasets library.
For the Bengali dataset, refer to the datasets/ directory for preprocessing scripts.


Run the Notebook:

Open BARTsumm.ipynb in Jupyter Notebook or JupyterLab.
Follow the notebook sections to:
Load and preprocess the dataset.
Fine-tune the BART model.
Evaluate performance using ROUGE, BLEU, METEOR, and BERTScore.
Generate summaries for new scientific articles.




Example Command for inference:
from transformers import BartForConditionalGeneration, BartTokenizer
model = BartForConditionalGeneration.from_pretrained("./models/bart_finetuned")
tokenizer = BartTokenizer.from_pretrained("./models/bart_finetuned")
input_text = "Your scientific article text here..."
inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=150, num_beams=4)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)



Results
The fine-tuned BART model achieved the following performance on the PubMed test set:

ROUGE-1 F1: 0.4092
ROUGE-2 F1: 0.1760
ROUGE-L F1: 0.2607
BLEU-4: 0.1208
METEOR: 0.2902
BERTScore F1: 0.8594 (average across selected summaries)

Qualitative analysis shows strong semantic alignment, with high BERTScore F1 scores (e.g., 0.9259 for "Sheehan's Syndrome"). The model excels in biomedical texts but may require further optimization for long documents or highly technical content.
Future Work

Graph Transformer Integration: Combine SciBERT with Graph Transformer Networks to capture document structure and improve long-text summarization.
Bengali Summarization: Expand the Bengali dataset and apply cross-lingual transfer learning with models like XLM-RoBERTa.
Real-World Deployment: Optimize for low-latency inference and integrate into academic or healthcare systems.
Advanced Techniques: Explore controllable summarization, knowledge-augmented summarization, and multimodal inputs.

Citation
If you use this code or dataset, please cite the thesis:
A. S. M. Montashir Fahim, "BARTSumm: Bidirectional Encoder & Autoregressive Decoder Transformer-based Scientific Text Abstractive Summarization," Bachelor of Science Thesis, Department of Computer Science & Engineering, Rajshahi University of Engineering & Technology, 2025.

Acknowledgments
This work was supervised by Prof. Dr. Boshir Ahmed, Department of Computer Science & Engineering, RUET. The project utilized resources from Kaggle and the Hugging Face ecosystem.
License
This project is licensed under the MIT License. See the LICENSE file for details.

For questions or contributions, please open an issue or contact the author at <montashirfahim25@gmail.com>.
