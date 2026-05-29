"""
Text Classification Module
Based on Fake-News-Detection framework

Classifies paper titles and abstracts for credibility and relevance
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Text classification result"""
    text: str
    class_label: str
    confidence: float
    explanation: str


class FakeNewsDetector:
    """
    Detects potentially unreliable papers/content in DBpia data

    Use cases:
    - Validate paper titles and abstracts
    - Identify clickbait-style papers
    - Flag suspicious references
    - Assess paper quality indicators
    """

    def __init__(self):
        self.logger = logger
        self.clickbait_keywords = {
            'shocking': 0.8,
            'revealed': 0.7,
            'hidden': 0.7,
            'amazing': 0.6,
            'unbelievable': 0.8,
            'doctors hate': 0.9,
            'government doesnt want': 0.9,
            'exclusive': 0.5,
            'you wont believe': 0.8,
        }
        self.quality_indicators = {
            'doi': 0.3,
            'peer review': 0.4,
            'conference': 0.3,
            'volume': 0.2,
            'pages': 0.2,
        }

    def classify_title(self, title: str) -> ClassificationResult:
        """
        Classify paper title for reliability

        Args:
            title: Paper title

        Returns:
            Classification result with confidence and explanation
        """
        title_lower = title.lower()

        # Clickbait check
        clickbait_score = self._check_clickbait(title_lower)

        # Quality indicators check
        quality_score = self._check_quality_indicators(title_lower)

        # Length check (too short = suspicious)
        length_score = self._check_title_length(title)

        # Grammar/coherence check (simplified)
        grammar_score = self._check_grammar(title)

        final_score = (
            clickbait_score * 0.4 +
            (1 - quality_score) * 0.3 +
            (1 - length_score) * 0.2 +
            (1 - grammar_score) * 0.1
        )

        if final_score > 0.7:
            label = 'suspicious'
        elif final_score > 0.4:
            label = 'questionable'
        else:
            label = 'reliable'

        explanation = self._generate_explanation(
            title, clickbait_score, quality_score, length_score, grammar_score
        )

        return ClassificationResult(
            text=title,
            class_label=label,
            confidence=min(1.0, final_score),
            explanation=explanation
        )

    def classify_abstract(self, abstract: str) -> ClassificationResult:
        """
        Classify paper abstract for credibility

        Args:
            abstract: Paper abstract

        Returns:
            Classification result
        """
        abstract_lower = abstract.lower()

        # Check for evidence-based language
        evidence_score = self._check_evidence_based(abstract_lower)

        # Check for proper citations format
        citation_score = self._check_citations(abstract)

        # Check for methodological rigor
        rigor_score = self._check_methodological_rigor(abstract_lower)

        # Check for overclaiming
        overclaim_score = self._check_overclaiming(abstract_lower)

        final_score = (
            evidence_score * 0.3 +
            citation_score * 0.2 +
            rigor_score * 0.3 +
            (1 - overclaim_score) * 0.2
        )

        if final_score > 0.7:
            label = 'credible'
        elif final_score > 0.4:
            label = 'neutral'
        else:
            label = 'questionable'

        return ClassificationResult(
            text=abstract[:100] + '...' if len(abstract) > 100 else abstract,
            class_label=label,
            confidence=min(1.0, final_score),
            explanation=f"Abstract credibility assessment: {label}"
        )

    def batch_classify(self, texts: List[str], mode: str = 'title') -> List[ClassificationResult]:
        """
        Classify multiple texts

        Args:
            texts: List of text samples
            mode: 'title' or 'abstract'

        Returns:
            List of classification results
        """
        classifier = self.classify_title if mode == 'title' else self.classify_abstract
        results = [classifier(text) for text in texts]

        suspicious_count = sum(1 for r in results if r.class_label == 'suspicious')
        self.logger.info(
            f"Classified {len(results)} texts, "
            f"found {suspicious_count} suspicious items"
        )
        return results

    def _check_clickbait(self, text: str) -> float:
        """Check for clickbait patterns"""
        score = 0.0
        for keyword, weight in self.clickbait_keywords.items():
            if keyword in text:
                score = max(score, weight)
        return score

    def _check_quality_indicators(self, text: str) -> float:
        """Check for quality indicators"""
        score = 0.0
        for indicator, weight in self.quality_indicators.items():
            if indicator in text:
                score += weight
        return min(1.0, score)

    def _check_title_length(self, title: str) -> float:
        """Check if title length is reasonable"""
        length = len(title.split())
        if 3 <= length <= 20:
            return 0.0
        elif 1 <= length <= 3 or 20 < length <= 50:
            return 0.3
        else:
            return 1.0

    def _check_grammar(self, text: str) -> float:
        """Simple grammar check"""
        # Very basic: check for capitalization, punctuation
        if not text[0].isupper():
            return 0.3
        if text.count('!!!') > 1 or text.count('???') > 1:
            return 0.5
        return 0.0

    def _check_evidence_based(self, abstract: str) -> float:
        """Check for evidence-based language"""
        evidence_keywords = ['study', 'research', 'analysis', 'found', 'showed', 'results', 'data']
        matches = sum(1 for keyword in evidence_keywords if keyword in abstract)
        return min(1.0, matches / 3)

    def _check_citations(self, text: str) -> float:
        """Check for proper citation format"""
        if re.search(r'\[\d+\]|\(\w+ \d{4}\)', text):
            return 1.0
        return 0.5

    def _check_methodological_rigor(self, text: str) -> float:
        """Check for methodological rigor indicators"""
        rigor_keywords = ['methodology', 'sample', 'statistical', 'validation', 'experiment', 'hypothesis']
        matches = sum(1 for keyword in rigor_keywords if keyword in text)
        return min(1.0, matches / 3)

    def _check_overclaiming(self, text: str) -> float:
        """Check for overclaiming language"""
        overclaim_keywords = ['cure', 'proven', 'definitely', 'always', 'never', 'guaranteed']
        matches = sum(1 for keyword in overclaim_keywords if keyword in text)
        return min(1.0, matches / 2)

    def _generate_explanation(self, title: str, *scores) -> str:
        """Generate human-readable explanation"""
        factors = []
        if scores[0] > 0.5:
            factors.append("clickbait-style language detected")
        if scores[1] < 0.3:
            factors.append("missing quality indicators")
        if scores[2] > 0.3:
            factors.append("unusual title length")
        if scores[3] > 0.3:
            factors.append("grammar/formatting issues")

        if factors:
            return f"Issues found: {', '.join(factors)}"
        return "Title appears reliable based on content analysis"
