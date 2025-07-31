import re
from typing import List, Tuple, Dict
from dataclasses import dataclass
import string
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
from pathlib import Path
import logging
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import pdfplumber

# Set NLTK data path if needed
nltk.data.path.append('D:\\project_apps\\PycharmProjects\\RAGenius\\models')

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = "embeddings_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


@dataclass
class ModelConfig:
    sequence_length: int = 32
    embedding_dim: int = 512
    hidden_dim: int = 512
    min_word_freq: int = 2
    batch_size: int = 16
    learning_rate: float = 0.001
    min_similarity_threshold: float = 0.15
    min_response_length: int = 20
    max_response_length: int = 200


class DocumentProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english', ngram_range=(1, 3),
            min_df=2, max_df=0.9, max_features=20000
        )
        self.documents = []
        self.document_vectors = None
        self.is_fitted = False
        self.min_similarity_threshold = 0.15
        self.segments = []
        self.segment_metadata = []

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'\([^)]*\d+[^)]*\)', '', text)
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\b\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\d+(\.\d+)*\s+(?=[A-Z])', '', text, flags=re.MULTILINE)
        text = re.sub(r'EXERCISES?\s*\d+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(?:Bibliography|References):.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        return text.strip()

    def segment_documents(self) -> List[str]:
        segments, metadata = [], []
        for doc_idx, doc in enumerate(self.documents):
            paragraphs = [p.strip() for p in doc.split('\n\n') if p.strip()]
            for para_idx, para in enumerate(paragraphs):
                para = self.clean_text(para)
                if len(para.split()) < 10:
                    continue
                if len(para.split()) > 100:
                    sentences = sent_tokenize(para)
                    current_segment, current_length = [], 0
                    for sent_idx, sentence in enumerate(sentences):
                        words = word_tokenize(sentence)
                        if current_length + len(words) <= 100:
                            current_segment.append(sentence)
                            current_length += len(words)
                        else:
                            if current_segment:
                                segment_text = ' '.join(current_segment)
                                segments.append(segment_text)
                                metadata.append({
                                    'doc_idx': doc_idx, 'para_idx': para_idx,
                                    'sent_idx': sent_idx, 'context': self._get_context(doc, para)
                                })
                            current_segment = [sentence]
                            current_length = len(words)
                    if current_segment:
                        segment_text = ' '.join(current_segment)
                        segments.append(segment_text)
                        metadata.append({
                            'doc_idx': doc_idx, 'para_idx': para_idx,
                            'sent_idx': len(sentences) - 1, 'context': self._get_context(doc, para)
                        })
                else:
                    segments.append(para)
                    metadata.append({
                        'doc_idx': doc_idx, 'para_idx': para_idx,
                        'sent_idx': 0, 'context': self._get_context(doc, para)
                    })
        self.segment_metadata = metadata
        return [seg for seg in segments if len(seg.split()) >= 10]

    def _get_context(self, doc: str, para: str) -> str:
        try:
            doc_lines = doc.split('\n\n')
            para_idx = doc_lines.index(para)
            context = ' '.join(doc_lines[max(0, para_idx - 1):min(len(doc_lines), para_idx + 2)])
            return self.clean_text(context)
        except:
            return ""

    def retrieve(self, query: str, top_k: int = 8) -> List[Tuple[str, float, dict]]:
        if not self.is_fitted:
            return []
        try:
            query = self.clean_text(query)
            query_vector = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
            matches, seen_content, unique_matches = [], set(), []

            for idx, sim in enumerate(similarities):
                if sim > self.min_similarity_threshold:
                    matches.append((self.segments[idx], sim, self.segment_metadata[idx]))

            matches.sort(key=lambda x: x[1], reverse=True)
            for content, sim, metadata in matches:
                norm_content = ' '.join(content.lower().split())
                if not any(self.is_similar_content(norm_content, seen) for seen in seen_content):
                    unique_matches.append((content, sim, metadata))
                    seen_content.add(norm_content)
                    if len(unique_matches) == top_k:
                        break
            return unique_matches
        except Exception as e:
            logger.error(f"Retrieval error: {str(e)}")
            return []

    def is_similar_content(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        if abs(len(text1) - len(text2)) / max(len(text1), len(text2)) > 0.2:
            return False
        words1 = set(word_tokenize(text1.lower())) - set(stopwords.words('english'))
        words2 = set(word_tokenize(text2.lower())) - set(stopwords.words('english'))
        return len(words1.intersection(words2)) / len(words1.union(words2)) > threshold if words1.union(
            words2) else False

    def _read_pdf(self, file_obj) -> str:
        try:
            with pdfplumber.open(file_obj) as pdf:
                return ' . '.join(page.extract_text() for page in pdf.pages if page.extract_text())
        except Exception as e:
            logger.error(f"PDF reading error: {str(e)}")
            return ""

    def _read_docx(self, file_path: Path) -> str:
        try:
            doc = docx.Document(file_path)
            return ' '.join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            logger.error(f"DOCX reading error: {str(e)}")
            return ""

    def _read_txt(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                return file_path.read_text(encoding='latin-1')
            except Exception as e:
                logger.error(f"Text file reading error: {str(e)}")
                return ""

    def upload_from_directory(self, directory: Path) -> None:
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        files = [f for ext in ['txt', 'pdf', 'docx'] for f in
                 list(directory.glob(f'*.{ext}')) + list(directory.glob(f'*.{ext.upper()}'))]
        processed_docs = []
        for file_path in files:
            try:
                if file_path.suffix.lower() == '.pdf':
                    text = self._read_pdf(file_path)
                elif file_path.suffix.lower() == '.docx':
                    text = self._read_docx(file_path)
                else:
                    text = self._read_txt(file_path)
                if text:
                    processed_docs.append(text)
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
        if processed_docs:
            self.upload_documents(processed_docs)

    def upload_documents(self, documents: List[str]) -> None:
        processed = [self.clean_text(doc) for doc in documents if doc and len(doc.split()) >= 10]
        self.documents = processed
        segments = self.segment_documents()
        self.document_vectors = self.vectorizer.fit_transform(segments)
        self.segments = segments
        self.is_fitted = True


class Chatbot:
    def __init__(self, sequence_length: int = 32):
        self.config = ModelConfig()
        self.document_processor = DocumentProcessor()
        self.sequence_length = sequence_length
        self.question_patterns = {
            'definition': r'what\s+(?:is|are)|define|explain|describe|meaning\s+of',
            'comparison': r'compare|difference|similar|versus|vs\.',
            'process': r'how\s+(?:do|does|can|could|would|should)|steps|process|procedure',
            'example': r'example|instance|illustrate',
            'analysis': r'why|analyze|reason|cause|effect|impact'
        }

    def identify_question_type(self, query: str) -> str:
        query = query.lower()
        for q_type, pattern in self.question_patterns.items():
            if re.search(pattern, query):
                return q_type
        return 'general'

    def generate_response(self, query: str) -> str:
        question_type = self.identify_question_type(query)
        chunks = self.document_processor.retrieve(query, top_k=8)
        if not chunks:
            return "I don't have enough information in the provided documents to answer that question."

        if question_type == 'definition':
            return self._generate_definition_response(chunks)
        elif question_type == 'comparison':
            return self._generate_comparison_response(chunks)
        elif question_type == 'process':
            return self._generate_process_response(chunks)
        elif question_type == 'example':
            return self._generate_example_response(chunks)
        elif question_type == 'analysis':
            return self._generate_analysis_response(chunks)
        else:
            return self._generate_general_response(chunks)

    def _generate_definition_response(self, chunks):
        response = "Definition:\n"
        response += '\n'.join([f"- {point}" for chunk in chunks for point in chunk[0].split('.') if point.strip()])
        return response

    def _generate_comparison_response(self, chunks):
        response = "Comparison:\n"
        response += '\n'.join([f"- {point}" for chunk in chunks[:2] for point in chunk[0].split('.') if point.strip()])
        return response

    def _generate_process_response(self, chunks):
        response = "Steps or Procedure:\n"
        response += '\n'.join([f"{i + 1}. {point}" for i, chunk in enumerate(chunks[:2]) for point in
                                chunk[0].split('.') if point.strip()])
        return response

    def _generate_example_response(self, chunks):
        examples = [chunk[0] for chunk in chunks if 'example' in chunk[0].lower()]
        if not examples:
            examples = [chunk[0] for chunk in chunks[:2]]
        response = "Here are some examples:\n"
        response += '\n'.join([f"- {ex}" for ex in examples])
        return response

    def _generate_analysis_response(self, chunks):
        causes_effects = [chunk[0] for chunk in chunks if
                          any(word in chunk[0].lower() for word in ['because', 'due to', 'cause', 'effect', 'impact'])]
        if not causes_effects:
            causes_effects = [chunk[0] for chunk in chunks[:2]]
        response = "Here's an analysis:\n"
        response += '\n'.join([f"- {ex}" for ex in causes_effects])
        return response

    def _generate_general_response(self, chunks):
        response = '\n'.join([f"- {chunk[0]}" for chunk in chunks[:3]])
        return response
