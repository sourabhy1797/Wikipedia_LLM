# **Wikipedia-Based Minimal Language Model: Training & Evaluation**

## **Project Overview**
This project fine-tunes **FLAN-T5** and **BART** on Wikipedia data to generate coherent responses. Additionally, an **ensemble model** was developed to combine the strengths of both models. The final models were evaluated using **ROUGE, BLEU, and METEOR scores** to measure their effectiveness.

## **Directory Structure**
```
|-- bart.py              # Fine-tunes the BART model
|-- flant5.py            # Fine-tunes the FLAN-T5 model
|-- ensemble.py          # Combines BART and FLAN-T5 predictions
|-- evaluate.py          # Computes ROUGE, BLEU, METEOR scores
|-- generate_answer.py   # Generates responses from trained models
```

---

## **Setup Instructions**

### **1. Install Dependencies**
Make sure you have all necessary dependencies installed.
```bash
pip install torch transformers datasets evaluate rouge_score nltk sacrebleu
```

### **2. Train Individual Models**
Run the following scripts to train the models:

**BART Training:**
```bash
python bart.py
```
**FLAN-T5 Training:**
```bash
python flant5.py
```

### **3. Run the Ensemble Model**
Once both models are trained, run:
```bash
python ensemble.py
```

### **4. Evaluate the Models**
To compute evaluation metrics:
```bash
python evaluate.py
```

### **5. Generate Sample Answers**
To test the model on a query:
```bash
python generate_answer.py
```

---

## **Model Choices and Justification**
- **FLAN-T5:** Strong at following instructions and structuring responses.
- **BART:** Good at generating fluent and paraphrased text.
- **Ensemble:** Selects words with the highest confidence from both models for balanced output.

---

## **Evaluation Metrics**
- **ROUGE Score**: Measures text overlap between generated and reference texts.
- **BLEU Score**: Evaluates precision at different n-gram levels.
- **METEOR Score**: Accounts for meaning preservation and synonyms.

**Results Comparison:**
| Model     | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU  | METEOR |
|-----------|--------|--------|--------|------|--------|
| FLAN-T5   | 0.1587 | 0.0322 | 0.1269 | 0.0  | 0.1974 |
| BART      | 0.0854 | 0.0172 | 0.0854 | 0.0  | 0.1369 |
| Ensemble  | 0.0584 | 0.0176 | 0.0584 | 0.0  | 0.1085 |

---

## **Future Enhancements**
- Implement **RAG (Retrieval-Augmented Generation)** to improve factual accuracy.
- Optimize **inference speed** by distilling models further.
- Explore **larger datasets** and more training epochs for fine-tuning.

