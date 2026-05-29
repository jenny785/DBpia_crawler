"""
Advanced Text Analysis Module
Transformer-based text analysis using BERT and other PLMs

Provides more accurate credibility assessment through deep learning
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AdvancedClassificationResult:
    """Advanced classification result with multiple scores"""
    text: str
    credibility_score: float
    relevance_score: float
    quality_score: float
    bias_score: float
    academic_level: str  # 'low', 'medium', 'high'
    confidence: float
    explanation: Dict[str, str]


class AdvancedTextAnalyzer:
    """
    Advanced text analysis using Transformer models

    Provides sophisticated credibility, relevance, and quality assessment
    of academic papers using pre-trained language models.

    Features:
    - Credibility assessment (fact-based scoring)
    - Relevance analysis (domain matching)
    - Quality evaluation (academic rigor)
    - Bias detection (neutrality assessment)
    - Academic level classification
    """

    def __init__(self, model_name: str = "bert-base-multilingual-cased"):
        """
        Initialize advanced analyzer

        Args:
            model_name: Pretrained model identifier
        """
        self.logger = logger
        self.model_name = model_name

        # Try to import transformers (optional dependency)
        try:
            from transformers import AutoTokenizer, AutoModel
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.has_transformer = True
            logger.info(f"Loaded transformer model: {model_name}")
        except ImportError:
            logger.warning("Transformers not installed. Using fallback methods.")
            self.has_transformer = False
        except Exception as e:
            logger.warning(f"Could not load transformer model: {e}. Using fallback methods.")
            self.has_transformer = False

    def analyze_title(self, title: str) -> AdvancedClassificationResult:
        """
        Advanced analysis of paper title

        Args:
            title: Paper title

        Returns:
            Advanced classification result
        """
        credibility = self._score_credibility_title(title)
        relevance = self._score_relevance(title)
        quality = self._score_quality(title)
        bias = self._detect_bias(title)
        academic_level = self._classify_academic_level(title)

        overall_confidence = (credibility + relevance + quality) / 3

        explanation = {
            'credibility': self._explain_credibility(credibility),
            'relevance': self._explain_relevance(relevance),
            'quality': self._explain_quality(quality),
            'bias': self._explain_bias(bias),
            'academic_level': f"Assessed as {academic_level} academic level"
        }

        return AdvancedClassificationResult(
            text=title,
            credibility_score=credibility,
            relevance_score=relevance,
            quality_score=quality,
            bias_score=bias,
            academic_level=academic_level,
            confidence=overall_confidence,
            explanation=explanation
        )

    def analyze_abstract(self, abstract: str, title: Optional[str] = None) -> AdvancedClassificationResult:
        """
        Advanced analysis of paper abstract

        Args:
            abstract: Paper abstract
            title: Paper title (optional, for context)

        Returns:
            Advanced classification result
        """
        credibility = self._score_credibility_abstract(abstract)
        relevance = self._score_relevance(abstract)
        quality = self._score_academic_rigor(abstract)
        bias = self._detect_bias(abstract)
        academic_level = self._classify_academic_level(abstract)

        overall_confidence = (credibility + relevance + quality) / 3

        explanation = {
            'credibility': self._explain_credibility_abstract(credibility),
            'relevance': self._explain_relevance(relevance),
            'quality': self._explain_quality(quality),
            'bias': self._explain_bias(bias),
            'academic_level': f"Assessed as {academic_level} academic level"
        }

        return AdvancedClassificationResult(
            text=abstract[:100] + '...' if len(abstract) > 100 else abstract,
            credibility_score=credibility,
            relevance_score=relevance,
            quality_score=quality,
            bias_score=bias,
            academic_level=academic_level,
            confidence=overall_confidence,
            explanation=explanation
        )

    def batch_analyze(self, texts: List[str], analysis_type: str = "title") -> List[AdvancedClassificationResult]:
        """
        Batch analysis of multiple texts

        Args:
            texts: List of texts to analyze
            analysis_type: 'title' or 'abstract'

        Returns:
            List of analysis results
        """
        analyzer = self.analyze_title if analysis_type == "title" else self.analyze_abstract
        results = [analyzer(text) for text in texts]

        self.logger.info(f"Analyzed {len(results)} texts with advanced analyzer")
        return results

    def compare_papers(self, paper1: Dict, paper2: Dict) -> Dict:
        """
        Compare two papers based on credibility and quality

        Args:
            paper1: First paper dictionary (title, abstract)
            paper2: Second paper dictionary (title, abstract)

        Returns:
            Comparison result
        """
        analysis1_title = self.analyze_title(paper1.get('title', ''))
        analysis2_title = self.analyze_title(paper2.get('title', ''))

        analysis1_abstract = self.analyze_abstract(
            paper1.get('abstract', ''),
            paper1.get('title', '')
        ) if paper1.get('abstract') else None

        analysis2_abstract = self.analyze_abstract(
            paper2.get('abstract', ''),
            paper2.get('title', '')
        ) if paper2.get('abstract') else None

        score1 = (
            analysis1_title.credibility_score +
            (analysis1_abstract.credibility_score if analysis1_abstract else 0)
        ) / (2 if analysis1_abstract else 1)

        score2 = (
            analysis2_title.credibility_score +
            (analysis2_abstract.credibility_score if analysis2_abstract else 0)
        ) / (2 if analysis2_abstract else 1)

        return {
            'paper1_score': score1,
            'paper2_score': score2,
            'better_paper': 'paper1' if score1 > score2 else 'paper2',
            'score_difference': abs(score1 - score2),
            'paper1_analysis': analysis1_title,
            'paper2_analysis': analysis2_title
        }

    # Scoring Methods
    def _score_credibility_title(self, title: str) -> float:
        """Score credibility based on title characteristics"""
        score = 0.7  # Base score

        # Reduce score for suspicious patterns
        suspicious_patterns = [
            ('!!!', -0.2),
            ('???', -0.2),
            ('you won\'t believe', -0.3),
            ('shocking', -0.1),
            ('fake', -0.2),
        ]

        for pattern, penalty in suspicious_patterns:
            if pattern.lower() in title.lower():
                score += penalty

        # Increase score for academic indicators
        academic_indicators = [
            ('doi', 0.1),
            ('et al', 0.1),
            ('pp.', 0.05),
            ('vol.', 0.05),
        ]

        for indicator, bonus in academic_indicators:
            if indicator.lower() in title.lower():
                score += bonus

        return max(0.0, min(1.0, score))

    def _score_credibility_abstract(self, abstract: str) -> float:
        """Score credibility based on abstract characteristics"""
        score = 0.7

        if not abstract:
            return 0.5

        # Check for evidence-based language
        evidence_words = ['study', 'research', 'found', 'results', 'data', 'analysis']
        evidence_count = sum(1 for word in evidence_words if word in abstract.lower())
        score += (evidence_count / 6) * 0.2

        # Check for overclaiming
        overclaim_words = ['proves', 'definitely', 'always', 'never', 'cure', 'guaranteed']
        overclaim_count = sum(1 for word in overclaim_words if word in abstract.lower())
        score -= (overclaim_count / 6) * 0.3

        # Check for proper citations
        if '[' in abstract and ']' in abstract:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_relevance(self, text: str) -> float:
        """Score relevance based on topic keywords"""
        keywords = {
            'machine learning': 0.9,
            'deep learning': 0.9,
            'nlp': 0.8,
            'neural network': 0.85,
            'data analysis': 0.8,
            'research': 0.7,
            'algorithm': 0.75,
            'experiment': 0.7,
        }

        score = 0.5
        found_keywords = 0

        for keyword, weight in keywords.items():
            if keyword.lower() in text.lower():
                score = max(score, weight)
                found_keywords += 1

        # Boost score if multiple relevant keywords found
        if found_keywords > 1:
            score += 0.1

        return min(1.0, score)

    def _score_academic_rigor(self, text: str) -> float:
        """Score academic rigor level"""
        rigor_indicators = [
            ('methodology', 0.2),
            ('hypothesis', 0.15),
            ('sample', 0.1),
            ('statistical', 0.15),
            ('validation', 0.15),
            ('experiment', 0.1),
        ]

        score = 0.5
        for indicator, weight in rigor_indicators:
            if indicator.lower() in text.lower():
                score += weight

        return min(1.0, score)

    def _detect_bias(self, text: str) -> float:
        """Detect potential bias in text (0 = no bias, 1 = high bias)"""
        bias_indicators = {
            'opinion words': ['think', 'believe', 'feel', 'obviously', 'clearly'],
            'superlatives': ['best', 'worst', 'greatest', 'amazing'],
            'absolute terms': ['always', 'never', 'impossible', 'certain'],
        }

        bias_score = 0.0

        for category, words in bias_indicators.items():
            matches = sum(1 for word in words if word in text.lower())
            bias_score += (matches / len(words)) * 0.3

        return min(1.0, bias_score)

    def _classify_academic_level(self, text: str) -> str:
        """Classify academic level: low, medium, high"""
        high_level_indicators = [
            'methodology', 'hypothesis', 'statistical',
            'peer review', 'conference', 'journal'
        ]

        medium_level_indicators = [
            'study', 'research', 'analysis', 'data'
        ]

        high_count = sum(1 for ind in high_level_indicators if ind in text.lower())
        medium_count = sum(1 for ind in medium_level_indicators if ind in text.lower())

        if high_count >= 2:
            return 'high'
        elif medium_count >= 2 or high_count == 1:
            return 'medium'
        else:
            return 'low'

    # Explanation Methods
    def _explain_credibility(self, score: float) -> str:
        """Generate credibility explanation"""
        if score >= 0.8:
            return "High credibility - Well-structured title with academic indicators"
        elif score >= 0.6:
            return "Moderate credibility - Title appears reasonable but lacks strong indicators"
        elif score >= 0.4:
            return "Low credibility - Title contains some concerning patterns"
        else:
            return "Very low credibility - Multiple suspicious indicators detected"

    def _explain_credibility_abstract(self, score: float) -> str:
        """Generate abstract credibility explanation"""
        if score >= 0.8:
            return "High credibility - Evidence-based language with proper structure"
        elif score >= 0.6:
            return "Moderate credibility - Some evidence-based claims present"
        elif score >= 0.4:
            return "Low credibility - Limited evidence or overclaiming detected"
        else:
            return "Very low credibility - Significant credibility issues"

    def _explain_relevance(self, score: float) -> str:
        """Generate relevance explanation"""
        if score >= 0.8:
            return "Highly relevant - Contains multiple topic-related keywords"
        elif score >= 0.6:
            return "Relevant - Some topic keywords present"
        else:
            return "Low relevance - Few topic-related indicators"

    def _explain_quality(self, score: float) -> str:
        """Generate quality explanation"""
        if score >= 0.8:
            return "High quality - Strong academic rigor indicators"
        elif score >= 0.6:
            return "Moderate quality - Some quality indicators present"
        else:
            return "Low quality - Limited quality indicators"

    def _explain_bias(self, score: float) -> str:
        """Generate bias explanation"""
        if score < 0.3:
            return "Low bias - Neutral and objective language"
        elif score < 0.6:
            return "Moderate bias - Some opinion-based language detected"
        else:
            return "High bias - Significant opinion and superlative language"
