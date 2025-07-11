"""
Error classification and pattern recognition system.
"""

from typing import Dict, List, Optional, Pattern, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import re
import threading
from collections import defaultdict

from .error_manager import ErrorContext, ErrorSeverity, ErrorCategory, ErrorType
from utils.logging import get_logger

logger = get_logger(__name__)


class PatternType(Enum):
    """Types of error patterns."""
    REGEX = "regex"
    KEYWORD = "keyword"
    FREQUENCY = "frequency"
    SEQUENCE = "sequence"
    CORRELATION = "correlation"


@dataclass
class ErrorPattern:
    """Definition of an error pattern."""
    pattern_id: str
    name: str
    pattern_type: PatternType
    
    # Pattern definition
    pattern: str  # Regex pattern, keywords, etc.
    severity_threshold: ErrorSeverity = ErrorSeverity.MEDIUM
    frequency_threshold: int = 5  # Number of occurrences
    time_window: timedelta = field(default_factory=lambda: timedelta(minutes=15))
    
    # Classification rules
    target_category: Optional[ErrorCategory] = None
    target_type: Optional[ErrorType] = None
    confidence_score: Decimal = Decimal('0.8')
    
    # Actions
    auto_escalate: bool = False
    notify_channels: List[str] = field(default_factory=list)
    recovery_strategy: Optional[str] = None
    
    # Metadata
    description: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def matches(self, error_context: ErrorContext) -> bool:
        """Check if this pattern matches the given error context."""
        if self.pattern_type == PatternType.REGEX:
            return self._match_regex(error_context)
        elif self.pattern_type == PatternType.KEYWORD:
            return self._match_keyword(error_context)
        else:
            return False
    
    def _match_regex(self, error_context: ErrorContext) -> bool:
        """Match using regex pattern."""
        try:
            pattern = re.compile(self.pattern, re.IGNORECASE)
            
            # Check message
            if pattern.search(error_context.message):
                return True
            
            # Check stack trace if available
            if error_context.stack_trace and pattern.search(error_context.stack_trace):
                return True
            
            # Check exception string
            if error_context.exception and pattern.search(str(error_context.exception)):
                return True
            
            return False
            
        except re.error:
            logger.warning(f"Invalid regex pattern: {self.pattern}")
            return False
    
    def _match_keyword(self, error_context: ErrorContext) -> bool:
        """Match using keyword search."""
        keywords = self.pattern.lower().split(',')
        text_to_search = (
            error_context.message + " " +
            (error_context.stack_trace or "") + " " +
            (str(error_context.exception) if error_context.exception else "")
        ).lower()
        
        return any(keyword.strip() in text_to_search for keyword in keywords)


@dataclass
class ClassificationResult:
    """Result of error classification."""
    error_id: str
    confidence: Decimal
    matched_patterns: List[ErrorPattern]
    suggested_category: Optional[ErrorCategory] = None
    suggested_type: Optional[ErrorType] = None
    suggested_severity: Optional[ErrorSeverity] = None
    escalation_required: bool = False
    recovery_suggestions: List[str] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'error_id': self.error_id,
            'confidence': float(self.confidence),
            'matched_patterns': [p.pattern_id for p in self.matched_patterns],
            'suggested_category': self.suggested_category.value if self.suggested_category else None,
            'suggested_type': self.suggested_type.value if self.suggested_type else None,
            'suggested_severity': self.suggested_severity.value if self.suggested_severity else None,
            'escalation_required': self.escalation_required,
            'recovery_suggestions': self.recovery_suggestions,
            'notification_channels': self.notification_channels
        }


class ErrorClassifier:
    """Intelligent error classification system."""
    
    def __init__(self):
        self.patterns: Dict[str, ErrorPattern] = {}
        self.classification_history: List[ClassificationResult] = []
        self.pattern_performance: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            'matches': 0,
            'correct_classifications': 0,
            'false_positives': 0
        })
        
        # Learning parameters
        self.learning_enabled = True
        self.min_confidence_threshold = Decimal('0.6')
        
        # Statistics
        self.classification_stats = {
            'total_classifications': 0,
            'successful_classifications': 0,
            'pattern_matches': 0,
            'auto_escalations': 0
        }
        
        self._lock = threading.Lock()
        
        # Initialize with default patterns
        self._initialize_default_patterns()
        
        logger.info("Error classifier initialized")
    
    def classify_error(self, error_context: ErrorContext) -> ClassificationResult:
        """
        Classify an error using pattern matching and machine learning.
        
        Args:
            error_context: The error to classify
            
        Returns:
            ClassificationResult with classification details
        """
        with self._lock:
            result = ClassificationResult(
                error_id=error_context.error_id,
                confidence=Decimal('0.0'),
                matched_patterns=[]
            )
            
            # Pattern matching
            matched_patterns = self._match_patterns(error_context)
            result.matched_patterns = matched_patterns
            
            if matched_patterns:
                # Calculate confidence and suggestions
                self._calculate_classification_confidence(result, matched_patterns)
                self._generate_suggestions(result, matched_patterns)
                
                # Update statistics
                self.classification_stats['pattern_matches'] += 1
                
                # Check for escalation
                if any(pattern.auto_escalate for pattern in matched_patterns):
                    result.escalation_required = True
                    self.classification_stats['auto_escalations'] += 1
            
            # Fallback classification using heuristics
            if result.confidence < self.min_confidence_threshold:
                self._heuristic_classification(error_context, result)
            
            # Store classification result
            self.classification_history.append(result)
            self.classification_stats['total_classifications'] += 1
            
            if result.confidence >= self.min_confidence_threshold:
                self.classification_stats['successful_classifications'] += 1
            
            logger.debug(f"Classified error {error_context.error_id} with confidence {result.confidence}")
            
            return result
    
    def add_pattern(self, pattern: ErrorPattern) -> None:
        """Add a new error pattern."""
        with self._lock:
            self.patterns[pattern.pattern_id] = pattern
            logger.info(f"Added error pattern: {pattern.pattern_id}")
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """Remove an error pattern."""
        with self._lock:
            if pattern_id in self.patterns:
                del self.patterns[pattern_id]
                logger.info(f"Removed error pattern: {pattern_id}")
                return True
            return False
    
    def update_pattern(self, pattern: ErrorPattern) -> None:
        """Update an existing pattern."""
        with self._lock:
            if pattern.pattern_id in self.patterns:
                pattern.last_updated = datetime.now()
                self.patterns[pattern.pattern_id] = pattern
                logger.info(f"Updated error pattern: {pattern.pattern_id}")
    
    def get_pattern_performance(self, pattern_id: str) -> Dict[str, Any]:
        """Get performance metrics for a pattern."""
        with self._lock:
            if pattern_id not in self.patterns:
                return {}
            
            pattern = self.patterns[pattern_id]
            performance = self.pattern_performance[pattern_id]
            
            total_matches = performance['matches']
            accuracy = (
                performance['correct_classifications'] / total_matches
                if total_matches > 0 else 0
            )
            
            return {
                'pattern_id': pattern_id,
                'pattern_name': pattern.name,
                'total_matches': total_matches,
                'correct_classifications': performance['correct_classifications'],
                'false_positives': performance['false_positives'],
                'accuracy': accuracy,
                'confidence_score': float(pattern.confidence_score),
                'last_updated': pattern.last_updated.isoformat()
            }
    
    def get_classification_statistics(self) -> Dict[str, Any]:
        """Get classification statistics."""
        with self._lock:
            stats = self.classification_stats.copy()
            
            # Calculate derived metrics
            total = stats['total_classifications']
            if total > 0:
                stats['success_rate'] = stats['successful_classifications'] / total
                stats['pattern_match_rate'] = stats['pattern_matches'] / total
                stats['escalation_rate'] = stats['auto_escalations'] / total
            else:
                stats['success_rate'] = 0.0
                stats['pattern_match_rate'] = 0.0
                stats['escalation_rate'] = 0.0
            
            # Add pattern statistics
            stats['total_patterns'] = len(self.patterns)
            stats['active_patterns'] = len([p for p in self.patterns.values() if p.confidence_score >= self.min_confidence_threshold])
            
            return stats
    
    def learn_from_feedback(
        self,
        error_id: str,
        correct_category: ErrorCategory,
        correct_type: ErrorType,
        correct_severity: ErrorSeverity
    ) -> None:
        """Learn from human feedback to improve classification accuracy."""
        if not self.learning_enabled:
            return
        
        with self._lock:
            # Find the classification result
            classification = None
            for result in self.classification_history:
                if result.error_id == error_id:
                    classification = result
                    break
            
            if not classification:
                logger.warning(f"Classification result not found for error: {error_id}")
                return
            
            # Update pattern performance based on feedback
            for pattern in classification.matched_patterns:
                performance = self.pattern_performance[pattern.pattern_id]
                performance['matches'] += 1
                
                # Check if classification was correct
                correct_classification = (
                    classification.suggested_category == correct_category and
                    classification.suggested_type == correct_type and
                    classification.suggested_severity == correct_severity
                )
                
                if correct_classification:
                    performance['correct_classifications'] += 1
                else:
                    performance['false_positives'] += 1
                
                # Adjust pattern confidence based on performance
                self._adjust_pattern_confidence(pattern.pattern_id)
            
            logger.info(f"Learned from feedback for error: {error_id}")
    
    def export_patterns(self, file_path: str) -> None:
        """Export patterns to file."""
        import json
        
        with self._lock:
            pattern_data = {}
            for pattern_id, pattern in self.patterns.items():
                pattern_data[pattern_id] = {
                    'name': pattern.name,
                    'pattern_type': pattern.pattern_type.value,
                    'pattern': pattern.pattern,
                    'severity_threshold': pattern.severity_threshold.value,
                    'frequency_threshold': pattern.frequency_threshold,
                    'time_window_minutes': int(pattern.time_window.total_seconds() / 60),
                    'target_category': pattern.target_category.value if pattern.target_category else None,
                    'target_type': pattern.target_type.value if pattern.target_type else None,
                    'confidence_score': float(pattern.confidence_score),
                    'auto_escalate': pattern.auto_escalate,
                    'notify_channels': pattern.notify_channels,
                    'recovery_strategy': pattern.recovery_strategy,
                    'description': pattern.description,
                    'examples': pattern.examples
                }
            
            with open(file_path, 'w') as f:
                json.dump(pattern_data, f, indent=2)
            
            logger.info(f"Exported {len(pattern_data)} patterns to {file_path}")
    
    def import_patterns(self, file_path: str) -> int:
        """Import patterns from file."""
        import json
        
        try:
            with open(file_path, 'r') as f:
                pattern_data = json.load(f)
            
            imported_count = 0
            for pattern_id, data in pattern_data.items():
                pattern = ErrorPattern(
                    pattern_id=pattern_id,
                    name=data['name'],
                    pattern_type=PatternType(data['pattern_type']),
                    pattern=data['pattern'],
                    severity_threshold=ErrorSeverity(data['severity_threshold']),
                    frequency_threshold=data['frequency_threshold'],
                    time_window=timedelta(minutes=data['time_window_minutes']),
                    target_category=ErrorCategory(data['target_category']) if data.get('target_category') else None,
                    target_type=ErrorType(data['target_type']) if data.get('target_type') else None,
                    confidence_score=Decimal(str(data['confidence_score'])),
                    auto_escalate=data['auto_escalate'],
                    notify_channels=data['notify_channels'],
                    recovery_strategy=data.get('recovery_strategy'),
                    description=data.get('description'),
                    examples=data.get('examples', [])
                )
                
                self.add_pattern(pattern)
                imported_count += 1
            
            logger.info(f"Imported {imported_count} patterns from {file_path}")
            return imported_count
            
        except Exception as e:
            logger.error(f"Failed to import patterns from {file_path}: {e}")
            return 0
    
    # Private methods
    
    def _initialize_default_patterns(self) -> None:
        """Initialize with default error patterns."""
        default_patterns = [
            # Trading errors
            ErrorPattern(
                pattern_id="insufficient_funds",
                name="Insufficient Funds",
                pattern_type=PatternType.KEYWORD,
                pattern="insufficient funds,insufficient balance,not enough balance",
                target_category=ErrorCategory.TRADING,
                target_type=ErrorType.INSUFFICIENT_FUNDS,
                confidence_score=Decimal('0.95'),
                auto_escalate=True
            ),
            
            ErrorPattern(
                pattern_id="market_closed",
                name="Market Closed",
                pattern_type=PatternType.KEYWORD,
                pattern="market closed,market not open,trading halted",
                target_category=ErrorCategory.TRADING,
                target_type=ErrorType.MARKET_CLOSED,
                confidence_score=Decimal('0.90')
            ),
            
            # Network errors
            ErrorPattern(
                pattern_id="connection_timeout",
                name="Connection Timeout",
                pattern_type=PatternType.REGEX,
                pattern=r"(timeout|timed out|connection.*timeout)",
                target_category=ErrorCategory.NETWORK,
                target_type=ErrorType.CONNECTION_TIMEOUT,
                confidence_score=Decimal('0.85'),
                recovery_strategy="retry_with_backoff"
            ),
            
            ErrorPattern(
                pattern_id="api_rate_limit",
                name="API Rate Limit",
                pattern_type=PatternType.KEYWORD,
                pattern="rate limit,too many requests,quota exceeded",
                target_category=ErrorCategory.NETWORK,
                target_type=ErrorType.API_RATE_LIMIT,
                confidence_score=Decimal('0.90'),
                recovery_strategy="exponential_backoff"
            ),
            
            # Data errors
            ErrorPattern(
                pattern_id="missing_data",
                name="Missing Data",
                pattern_type=PatternType.KEYWORD,
                pattern="missing data,data not found,no data available",
                target_category=ErrorCategory.DATA,
                target_type=ErrorType.MISSING_DATA,
                confidence_score=Decimal('0.80')
            ),
            
            # Risk management errors
            ErrorPattern(
                pattern_id="risk_limit_breach",
                name="Risk Limit Breach",
                pattern_type=PatternType.KEYWORD,
                pattern="risk limit,limit exceeded,risk threshold",
                target_category=ErrorCategory.RISK_MANAGEMENT,
                target_type=ErrorType.RISK_LIMIT_BREACH,
                confidence_score=Decimal('0.95'),
                auto_escalate=True,
                severity_threshold=ErrorSeverity.HIGH
            )
        ]
        
        for pattern in default_patterns:
            self.patterns[pattern.pattern_id] = pattern
        
        logger.info(f"Initialized {len(default_patterns)} default patterns")
    
    def _match_patterns(self, error_context: ErrorContext) -> List[ErrorPattern]:
        """Find patterns that match the error context."""
        matched_patterns = []
        
        for pattern in self.patterns.values():
            if pattern.matches(error_context):
                matched_patterns.append(pattern)
        
        # Sort by confidence score
        matched_patterns.sort(key=lambda p: p.confidence_score, reverse=True)
        
        return matched_patterns
    
    def _calculate_classification_confidence(
        self,
        result: ClassificationResult,
        patterns: List[ErrorPattern]
    ) -> None:
        """Calculate classification confidence based on matched patterns."""
        if not patterns:
            result.confidence = Decimal('0.0')
            return
        
        # Use highest confidence pattern as base
        best_pattern = patterns[0]
        result.confidence = best_pattern.confidence_score
        
        # Adjust confidence based on pattern agreement
        if len(patterns) > 1:
            # If multiple patterns agree on category/type, increase confidence
            categories = [p.target_category for p in patterns if p.target_category]
            types = [p.target_type for p in patterns if p.target_type]
            
            if len(set(categories)) == 1 and len(categories) > 1:
                result.confidence = min(result.confidence * Decimal('1.2'), Decimal('1.0'))
            
            if len(set(types)) == 1 and len(types) > 1:
                result.confidence = min(result.confidence * Decimal('1.1'), Decimal('1.0'))
    
    def _generate_suggestions(
        self,
        result: ClassificationResult,
        patterns: List[ErrorPattern]
    ) -> None:
        """Generate classification suggestions based on patterns."""
        if not patterns:
            return
        
        # Use highest confidence pattern for suggestions
        best_pattern = patterns[0]
        
        result.suggested_category = best_pattern.target_category
        result.suggested_type = best_pattern.target_type
        result.suggested_severity = best_pattern.severity_threshold
        
        # Collect recovery suggestions
        for pattern in patterns:
            if pattern.recovery_strategy:
                result.recovery_suggestions.append(pattern.recovery_strategy)
            
            result.notification_channels.extend(pattern.notify_channels)
        
        # Remove duplicates
        result.recovery_suggestions = list(set(result.recovery_suggestions))
        result.notification_channels = list(set(result.notification_channels))
    
    def _heuristic_classification(
        self,
        error_context: ErrorContext,
        result: ClassificationResult
    ) -> None:
        """Perform heuristic classification when pattern matching fails."""
        message = error_context.message.lower()
        
        # Simple keyword-based classification
        if any(word in message for word in ['timeout', 'connection', 'network']):
            result.suggested_category = ErrorCategory.NETWORK
            result.suggested_type = ErrorType.CONNECTION_TIMEOUT
            result.confidence = Decimal('0.5')
        
        elif any(word in message for word in ['data', 'missing', 'not found']):
            result.suggested_category = ErrorCategory.DATA
            result.suggested_type = ErrorType.MISSING_DATA
            result.confidence = Decimal('0.5')
        
        elif any(word in message for word in ['order', 'trade', 'trading']):
            result.suggested_category = ErrorCategory.TRADING
            result.suggested_type = ErrorType.ORDER_REJECTION
            result.confidence = Decimal('0.4')
        
        elif any(word in message for word in ['config', 'configuration']):
            result.suggested_category = ErrorCategory.CONFIGURATION
            result.suggested_type = ErrorType.INVALID_CONFIG
            result.confidence = Decimal('0.4')
        
        else:
            result.suggested_category = ErrorCategory.SYSTEM
            result.confidence = Decimal('0.3')
    
    def _adjust_pattern_confidence(self, pattern_id: str) -> None:
        """Adjust pattern confidence based on performance feedback."""
        if pattern_id not in self.patterns:
            return
        
        pattern = self.patterns[pattern_id]
        performance = self.pattern_performance[pattern_id]
        
        total_matches = performance['matches']
        if total_matches < 5:  # Need minimum data points
            return
        
        accuracy = performance['correct_classifications'] / total_matches
        
        # Adjust confidence score based on accuracy
        if accuracy > 0.9:
            pattern.confidence_score = min(pattern.confidence_score * Decimal('1.05'), Decimal('1.0'))
        elif accuracy < 0.6:
            pattern.confidence_score = max(pattern.confidence_score * Decimal('0.95'), Decimal('0.1'))
        
        pattern.last_updated = datetime.now()


# Global error classifier instance
_error_classifier = None


def get_error_classifier() -> ErrorClassifier:
    """Get global error classifier instance."""
    global _error_classifier
    if _error_classifier is None:
        _error_classifier = ErrorClassifier()
    return _error_classifier