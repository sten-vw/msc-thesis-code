"""RAGAS: reference-free and reference-based RAG answer-quality metrics.

Implements RAGAS (Es et al., 2023, EACL 2024): scores the (question,
context, answer) triple with faithfulness, answer relevance and context
relevance, no ground truth needed. The reference-based extension adds
answer correctness and answer similarity against a gold or synthesized
reference. All scores are in [0, 1]. Prompts are verbatim from
explodinggradients/ragas.

Paper: https://arxiv.org/abs/2309.15217
"""

from __future__ import annotations

import re

import numpy as np

from core.llm import BedrockLLM

N_REVERSE_QUESTIONS = 3

# Source: ragas/metrics/_faithfulness.py — NLI_STATEMENTS_MESSAGE
STATEMENT_EXTRACTION_PROMPT = """\
Given a question and answer, create one or more statements from each sentence \
in the given answer.

question: {question}
answer: {answer}

Statements:"""

# Source: ragas/metrics/_faithfulness.py — NLI_INFERENCE_MESSAGE
NLI_VERIFICATION_PROMPT = """\
Consider the given context and following statements, then determine whether \
they are supported by the information present in the context. Provide a brief \
explanation for each statement before arriving at the verdict (Yes/No). \
Provide a final verdict for each statement in order at the end in the given \
format. Do not deviate from the specified format.

Context:
{context}

Statements:
{statements}

Answer in the following format for each statement:

Statement: [statement text]
Reason: [brief explanation]
Verdict: [Yes/No]"""

# Source: ragas/metrics/_answer_relevance.py — QUESTION_GEN
REVERSE_QUESTION_PROMPT = """\
Generate a question for the given answer.

answer: {answer}"""

# Source: ragas/metrics/_context_relevance.py — CONTEXT_RELEVANCE
CONTEXT_RELEVANCE_PROMPT = """\
Please extract relevant sentences from the provided context that can \
potentially help answer the following question. If no relevant sentences \
are found, or if you believe the question cannot be answered from the \
given context, return the phrase "Insufficient Information". While \
extracting candidate sentences you're not allowed to make any changes \
to sentences from given context.

question: {question}
context: {context}

Relevant sentences:"""

# explodinggradients/ragas CORRECTNESS_PROMPT (src/ragas/metrics/_answer_correctness.py)
ANSWER_CORRECTNESS_PROMPT = """\
Given a question, a ground truth answer, and a generated answer, classify \
each statement involved in answering the question into exactly one of \
three categories:

- TP (true positive): a statement present in the generated answer that is \
also supported by the ground truth answer.
- FP (false positive): a statement present in the generated answer that is \
not supported by the ground truth answer.
- FN (false negative): a statement that is required by the ground truth \
answer but missing from the generated answer.

Each statement belongs to exactly one category. You may briefly reason \
through the statements, but the LAST three lines of your response MUST be \
exactly these counts (integers, no other text on those lines):

TP: <count>
FP: <count>
FN: <count>

question: {question}
ground truth answer: {ground_truth}
generated answer: {answer}"""


class RagasJudge:
    """Reference-free faithfulness, answer relevance, context relevance, plus reference-based
    answer correctness and answer similarity.
    """

    def __init__(self, llm: BedrockLLM, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self.llm = llm
        self._embedding_model_name = embedding_model
        self._embedder = None

    def _get_embedder(self):
        """Lazy-load the sentence-transformers model for the embedding-based metrics."""
        if self._embedder is None:
            import os

            from sentence_transformers import SentenceTransformer

            device = os.environ.get("RAG_EVAL_ST_DEVICE") or None
            self._embedder = SentenceTransformer(self._embedding_model_name, device=device)
        return self._embedder

    def faithfulness(self, question: str, answer: str, context: str) -> float:
        """Decompose answer into claims, verify each against context; score = supported / total
        claims (0-1).
        """
        if not context:
            return 0.0

        prompt = STATEMENT_EXTRACTION_PROMPT.format(question=question, answer=answer)
        response = self.llm.invoke([BedrockLLM.user_message(prompt)], temperature=0.0)
        statements = _parse_statements(response.text)

        if not statements:
            return 0.0

        statements_text = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
        prompt = NLI_VERIFICATION_PROMPT.format(context=context, statements=statements_text)
        response = self.llm.invoke([BedrockLLM.user_message(prompt)], temperature=0.0)

        verdicts = _parse_verdicts(response.text, len(statements))
        supported = sum(verdicts)
        return supported / len(statements)

    def answer_relevance(self, question: str, answer: str) -> float:
        """Generate reverse questions from the answer; score = mean cosine similarity to the
        original question (0-1).
        """
        generated_questions = []
        for _ in range(N_REVERSE_QUESTIONS):
            prompt = REVERSE_QUESTION_PROMPT.format(answer=answer)
            response = self.llm.invoke([BedrockLLM.user_message(prompt)], temperature=0.7)
            q = response.text.strip()
            if q:
                generated_questions.append(q)

        if not generated_questions:
            return 0.0

        embedder = self._get_embedder()
        all_texts = [question] + generated_questions
        embeddings = embedder.encode(all_texts, normalize_embeddings=True, show_progress_bar=False)

        original_emb = embeddings[0]
        similarities = []
        for emb in embeddings[1:]:
            sim = float(np.dot(original_emb, emb))
            similarities.append(max(0.0, sim))

        return float(np.mean(similarities))

    def context_relevance(self, question: str, context: str) -> float:
        """Extract context sentences relevant to the question; score = extracted / total
        sentences (0-1).
        """
        if not context:
            return 0.0

        prompt = CONTEXT_RELEVANCE_PROMPT.format(question=question, context=context)
        response = self.llm.invoke([BedrockLLM.user_message(prompt)], temperature=0.0)

        total_sentences = len(_split_sentences(context))
        if total_sentences == 0:
            return 0.0

        extracted_text = response.text.strip()
        if "insufficient information" in extracted_text.lower():
            return 0.0

        extracted_sentences = len(_split_sentences(extracted_text))
        return min(1.0, extracted_sentences / total_sentences)

    def answer_correctness(self, question: str, answer: str, reference: str) -> float:
        """Claim-level F1 between answer and reference (RAGAS extension)."""
        if not answer.strip() or not reference.strip():
            return 0.0

        prompt = ANSWER_CORRECTNESS_PROMPT.format(
            question=question, ground_truth=reference, answer=answer
        )
        response = self.llm.invoke([BedrockLLM.user_message(prompt)], temperature=0.0)
        tp, fp, fn = _parse_tp_fp_fn(response.text)
        denom = tp + 0.5 * (fp + fn)
        if denom == 0:
            return 0.0
        return tp / denom

    def answer_similarity(self, answer: str, reference: str) -> float:
        """Cosine similarity between answer and reference embeddings."""
        if not answer.strip() or not reference.strip():
            return 0.0
        embedder = self._get_embedder()
        embs = embedder.encode(
            [answer, reference], normalize_embeddings=True, show_progress_bar=False,
        )
        sim = float(np.dot(embs[0], embs[1]))
        return max(0.0, sim)


def _parse_statements(text: str) -> list[str]:
    """Parse numbered or bulleted statements from LLM response."""
    statements = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[\d]+[.)]\s*", "", line)
        cleaned = re.sub(r"^[-*]\s*", "", cleaned)
        cleaned = cleaned.strip()
        if len(cleaned) > 5:
            statements.append(cleaned)
    return statements


def _parse_verdicts(text: str, n_statements: int) -> list[bool]:
    """Parse Verdict: Yes/No from NLI verification response."""
    verdicts = []
    for match in re.finditer(r"[Vv]erdict\s*:\s*(Yes|No)", text, re.IGNORECASE):
        verdicts.append(match.group(1).lower() == "yes")

    while len(verdicts) < n_statements:
        verdicts.append(False)

    return verdicts[:n_statements]


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if len(s.strip()) > 5]


def _parse_tp_fp_fn(text: str) -> tuple[int, int, int]:
    """Scans the whole response for `TP|FP|FN: <int>` and keeps the last occurrence of each."""
    counts = {"TP": 0, "FP": 0, "FN": 0}
    for label in counts:
        last_val: int | None = None
        for m in re.finditer(rf"\b{label}\s*:\s*(\d+)", text):
            last_val = int(m.group(1))
        if last_val is not None:
            counts[label] = last_val
    return counts["TP"], counts["FP"], counts["FN"]
