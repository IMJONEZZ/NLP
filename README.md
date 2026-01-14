# NLP - A Linguistic Approach

A collection of NLP code and reference materials organized by linguistic concepts rather than computational techniques.

## Quick Start

- **Python Version**: All code is Python 3.5+ compatible (tested through 3.8.x)
- **Utilities**: Shared helper functions in `utils/`
- **Datasets**: Common datasets in `data/`

## Why Linguistics?

> "Have you ever wondered why machine translation between English and Arabic or English and Japanese is uncharacteristically bad? Simply put, most people working on these systems have not known the intricacies of Alphabets vs Abjads vs Logo-Syllabaries. We're now at a point where we can apply mathematics to linguistics to improve specific languages. This requires a solid foundation in both parts."

## Topics

### [Writing Systems](./writing-systems/)
Orthographic foundations for understanding how different languages represent text
- Alphabets, Abjads, Abugidas, and Syllabaries

### [Foundations](./foundations/)
Mathematical and machine learning fundamentals for NLP
- Linear Algebra with NumPy
- Logistic Regression & Naive Bayes classifiers
- PCA, Hash Tables, and ML Primer

### [Morphology](./morphology/)
Word-level analysis and representation
- Bag of Words models
- Word2Vec embeddings
- Edit Distance (Levenshtein)

### [Syntax](./syntax/)
Structural analysis of language
- Part-of-Speech Tagging
- N-Grams and Autocomplete
- Tree Parsing

### [Semantics](./semantics/)
Meaning and sentiment analysis
- Sentiment Analysis (multiple implementations)
- Siamese Networks for semantic similarity

### [Translation](./translation/)
Cross-lingual NLP tasks
- Machine Translation (traditional & neural)
- Multilingual Classification
- Google Translate fine-tuning

### [Language Models](./language-models/)
Text generation and prediction
- Markov Chains
- RNNs, LSTMs, GRUs

### [Attention & Transformers](./attention-transformers/)
State-of-the-art architectures
- Attention mechanisms
- BERT, GPT-J, T5 fine-tuning projects
- Transformer implementations

### [Text Preprocessing](./text-preprocessing/)
Tokenization and normalization
- Byte Pair Encoding (BPE)
- SentencePiece

### [Applications](./applications/)
Practical NLP tasks
- Named Entity Recognition (NER)
- Question Answering
- Topic Modeling
- Text Analysis projects

### [Generation](./generation/)
Text and multimodal generation
- Chatbot with RevNet
- Stable Diffusion
- DALL-E

## Documentation

Course materials, slides, and reference papers are organized by topic in `docs/`:
- **Foundations**: Formula sheets, course slides on fundamentals
- **Morphology**: Word embedding papers and probabilistic models
- **Syntax**: POS tagging, n-grams documentation
- **Language Models**: Sequence model slides and notes
- **Attention Transformers**: Attention mechanism papers

## Publications

Accolades and published work are available in `[_publications/](_publications/)`

### Featured Publication
**Machine Learning Based Chat Analysis (Libchat)** - November 2020
- ML-based tool for analyzing chat transcripts between patrons and librarians
- Tasks: estimating patron satisfaction, classifying queries (Research/Reference, Directional, Tech/Troubleshooting, Policy/Procedure)
- Achieved 78%+ accuracy for each category
- Includes toy dataset

## Project Sources

Projects are adapted from multiple sources:
- **DeepLearning.ai NLP Specialization** (Stanford)
- **Codecademy NLP Course**
- **Custom implementations and research projects**

Each project folder preserves its original structure while being organized by linguistic topic. Source references are included in file headers where applicable.