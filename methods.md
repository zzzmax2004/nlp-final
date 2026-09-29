# Methods to try
- Dataset comes from StackExchange, we can fine tune a pre-trained SOTA model on it to improve understanding of technical Q&A.
- Pseudo-lableing
    - First pretriain a model on the labeled data
    - Then use the model to predict on the unlabeled data (data scraped from StackExchange)
    - Use these "pseudo-labels" to further train the model on combined dataset
    - Use ensemble of models to generate pseudo-labels to improve quality
    - Ensure no data leakage from labeled to unlabeled set (don't include test set )
- Head tail truncation
    - Currently, tokenizer truncated the first 512 tokens, but in technical Q&A, important information may be at the end (e.g. code snippets, error messages)
    - Experiment with different truncation strategies, e.g. keep last 512 tokens, or keep both head and tail (e.g. first 256 + last 256)

# Creative methods
- LLM-as-a-Judge
- Add sentiment analysis to data preprocessing as data aug
- Use readability score as a feature
- Question answer alignment score (embeddings, consine similarity)
- Exploit distilled knowledge
    - First we use a SOTA model to analyze the QA pairs and generate a reasoning verdict, eg. (this is good because....)
    - Then, instead of training on QA pairs, we train on these reasoning verdicts
- Question type classification
    - Appedn question type as a text token, e.g. [DEBUGGING], [CONCEPTUAL], [BEST_PRACTICES]