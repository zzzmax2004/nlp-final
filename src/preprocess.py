import pandas as pd
import re
import html
import json
from pathlib import Path


class QuestPreprocessor:
    def __init__(self, use_category=True, use_host=True, max_body_words=500):
        self.use_category = use_category
        self.use_host = use_host
        # Limit body to roughly 500 words (approx 700-800 tokens) to save room for Answer
        self.max_body_words = max_body_words
        
        # Common contractions to expand (Simplified version of the Inference notebook's dictionary)
        self.contractions = {
            "n't": " not", "'re": " are", "'s": " is", "'d": " would",
            "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
        }

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Apply cleaning to specific columns
        text_cols = ['question_title', 'question_body', 'answer']
        for col in text_cols:
            df[col] = df[col].astype(str).apply(self._clean_text)
            
        df['text_input'] = df.apply(self._construct_input_text, axis=1)
        return df

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str): return str(text)
        
        # 1. Decode HTML (Turn &gt; into >)
        text = html.unescape(text)
        
        # 2. Remove HTML Tags but keep structure hints like paragraphs
        text = re.sub(r'<br\s*/?>', ' ', text) # Replace breaks with space
        text = re.sub(r'<[^>]+>', '', text)    # Remove other tags
        
        # 3. Simplify Math (LaTeX) to a placeholder
        # This helps the tokenizer not get confused by complex equations
        text = re.sub(r'\$.*?\$', ' [MATH] ', text)
        
        # 4. Mask Numbers (From Inference Notebook)
        # Prevents overfitting to specific IDs or dates. 
        # 1234 -> ####
        text = re.sub(r'\d+', '#', text)
        
        # 5. Expand Contractions
        for contraction, expansion in self.contractions.items():
            text = text.replace(contraction, expansion)

        # 6. Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def _smart_truncate(self, text: str) -> str:
        words = text.split()
        if len(words) <= self.max_body_words:
            return text
        
        # Keep 50% Head, 50% Tail of the allowed limit
        half_limit = self.max_body_words // 2
        head = words[:half_limit]
        tail = words[-half_limit:]
        
        return " ".join(head) + " ... [TRUNCATED] ... " + " ".join(tail)

    def _construct_input_text(self, row) -> str:
        # Strategy: Metadata -> Title -> Body (Head+Tail) -> Answer
        # Moving Metadata to start ensures it is NEVER truncated.
        
        parts = []
        
        # 1. Metadata (Context)
        if self.use_category:
            parts.append(f"CAT: {str(row['category'])}")
        if self.use_host:
            parts.append(f"HOST: {str(row['host'])}")
            
        # 2. Title (High Priority)
        parts.append(f"TITLE: {str(row['question_title'])}")
        
        # 3. Body (Truncated with Head+Tail to save space)
        # We apply smart truncation here to ensure we leave room for the answer
        clean_body = str(row['question_body'])
        truncated_body = self._smart_truncate(clean_body)
        parts.append(f"BODY: {truncated_body}")
        
        # 4. Answer (High Priority - usually fits because Body was truncated)
        parts.append(f"ANSWER: {str(row['answer'])}")
        
        # Join with standard special token separator
        return " [SEP] ".join(parts)


class QuestPreprocessorWithVerdicts(QuestPreprocessor):
    """
    Extended preprocessor that incorporates Gemini-generated verdicts.
    
    The verdicts provide distilled knowledge about Q&A quality that helps
    the model learn what constitutes good questions and answers.
    """
    
    def __init__(self, verdicts_path: str = None, use_category=True, use_host=True, max_body_words=500):
        super().__init__(use_category, use_host, max_body_words)
        self.verdicts = {}
        
        if verdicts_path and Path(verdicts_path).exists():
            with open(verdicts_path, 'r') as f:
                self.verdicts = json.load(f)
            print(f"Loaded {len(self.verdicts)} verdicts from {verdicts_path}")
    
    def _construct_input_text(self, row) -> str:
        """Construct input text with Gemini verdicts prepended."""
        
        parts = []
        
        # 0. Gemini Verdict (Distilled Knowledge) - Prepend for maximum attention
        qa_id = str(row.get('qa_id', ''))
        if qa_id in self.verdicts:
            verdict_data = self.verdicts[qa_id]
            if isinstance(verdict_data, dict) and 'error' not in verdict_data:
                verdict_text = verdict_data.get('verdict', '')
                q_quality = verdict_data.get('q_quality', '')
                a_quality = verdict_data.get('a_quality', '')
                
                if verdict_text:
                    parts.append(f"VERDICT: {verdict_text}")
                if q_quality:
                    parts.append(f"Q_QUALITY: {q_quality}")
                if a_quality:
                    parts.append(f"A_QUALITY: {a_quality}")
        
        # 1. Metadata (Context)
        if self.use_category:
            parts.append(f"CAT: {str(row['category'])}")
        if self.use_host:
            parts.append(f"HOST: {str(row['host'])}")
            
        # 2. Title (High Priority)
        parts.append(f"TITLE: {str(row['question_title'])}")
        
        # 3. Body (Truncated with Head+Tail to save space)
        clean_body = str(row['question_body'])
        truncated_body = self._smart_truncate(clean_body)
        parts.append(f"BODY: {truncated_body}")
        
        # 4. Answer
        parts.append(f"ANSWER: {str(row['answer'])}")
        
        return " [SEP] ".join(parts)