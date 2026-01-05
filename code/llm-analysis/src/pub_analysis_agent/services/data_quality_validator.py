"""
Data Quality Validation Service.

This module provides comprehensive data quality checks and completeness validation
for assembled analysis results.
"""

import logging
from datetime import datetime, UTC
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from ..models.schema_validator import ConsolidatedAnalysisSchema
from ..services.dataset_service import DatasetService

logger = logging.getLogger(__name__)


class QualityLevel(str, Enum):
    """Quality levels for data validation."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class ValidationSeverity(str, Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class QualityIssue:
    """Represents a data quality issue."""
    
    field: str
    message: str
    severity: ValidationSeverity
    issue_type: str
    current_value: Any
    expected_value: Optional[Any] = None
    suggestion: Optional[str] = None


@dataclass
class CompletenessScore:
    """Represents completeness scoring for analysis results."""
    
    overall_score: float
    publication_metadata_score: float
    dataset_analysis_score: float
    code_extraction_score: float
    link_extraction_score: float
    workflow_metadata_score: float
    missing_critical_fields: List[str]
    missing_optional_fields: List[str]


@dataclass
class QualityMetrics:
    """Comprehensive quality metrics for analysis results."""
    
    completeness_score: CompletenessScore
    data_consistency_score: float
    logical_validation_score: float
    overall_quality_level: QualityLevel
    total_issues: int
    critical_issues: int
    warnings: int
    info_issues: int
    quality_issues: List[QualityIssue]
    validation_timestamp: datetime


class DataQualityValidator:
    """
    Comprehensive data quality validator for analysis results.
    
    This class provides extensive validation for completeness, consistency,
    and logical correctness of assembled analysis results.
    """
    
    def __init__(self, dataset_service: Optional[DatasetService] = None):
        """
        Initialize the data quality validator.
        
        Args:
            dataset_service: Optional DatasetService for dataset reference validation
        """
        self.dataset_service = dataset_service
        logger.info("DataQualityValidator initialized")
    
    async def validate_analysis_quality(
        self, 
        analysis_data: Dict[str, Any]
    ) -> QualityMetrics:
        """
        Perform comprehensive quality validation on analysis results.
        
        Args:
            analysis_data: Consolidated analysis data to validate
            
        Returns:
            QualityMetrics with comprehensive validation results
        """
        logger.info(f"Starting quality validation for publication {analysis_data.get('publication_id', 'unknown')}")
        
        quality_issues = []
        
        # Perform all validation checks
        quality_issues.extend(await self._validate_completeness(analysis_data))
        quality_issues.extend(await self._validate_data_consistency(analysis_data))
        quality_issues.extend(await self._validate_logical_rules(analysis_data))
        quality_issues.extend(await self._validate_dataset_references(analysis_data))
        quality_issues.extend(await self._validate_code_quality(analysis_data))
        quality_issues.extend(await self._validate_link_quality(analysis_data))
        
        # Calculate scores and metrics
        completeness_score = self._calculate_completeness_score(analysis_data, quality_issues)
        data_consistency_score = self._calculate_consistency_score(quality_issues)
        logical_validation_score = self._calculate_logical_score(quality_issues)
        overall_quality_level = self._determine_quality_level(completeness_score, quality_issues)
        
        # Count issues by severity
        critical_issues = len([i for i in quality_issues if i.severity == ValidationSeverity.CRITICAL])
        warnings = len([i for i in quality_issues if i.severity == ValidationSeverity.WARNING])
        info_issues = len([i for i in quality_issues if i.severity == ValidationSeverity.INFO])
        
        metrics = QualityMetrics(
            completeness_score=completeness_score,
            data_consistency_score=data_consistency_score,
            logical_validation_score=logical_validation_score,
            overall_quality_level=overall_quality_level,
            total_issues=len(quality_issues),
            critical_issues=critical_issues,
            warnings=warnings,
            info_issues=info_issues,
            quality_issues=quality_issues,
            validation_timestamp=datetime.now(UTC)
        )
        
        logger.info(f"Quality validation completed. Overall quality: {overall_quality_level}")
        return metrics
    
    async def _validate_completeness(self, analysis_data: Dict[str, Any]) -> List[QualityIssue]:
        """Validate completeness of analysis data."""
        issues = []
        
        # Check critical required fields
        critical_fields = [
            "publication_id", "analysis_timestamp", "workflow_status",
            "publication_metadata", "analysis_flags", "dataset_analysis",
            "code_extraction", "link_extraction", "workflow_metadata"
        ]
        
        for field in critical_fields:
            if field not in analysis_data or analysis_data[field] is None:
                issues.append(QualityIssue(
                    field=field,
                    message=f"Critical field '{field}' is missing",
                    severity=ValidationSeverity.CRITICAL,
                    issue_type="missing_critical_field",
                    current_value=None,
                    expected_value="present"
                ))
        
        # Check publication metadata completeness
        if "publication_metadata" in analysis_data:
            pub_meta = analysis_data["publication_metadata"]
            if isinstance(pub_meta, dict):
                if not pub_meta.get("title"):
                    issues.append(QualityIssue(
                        field="publication_metadata.title",
                        message="Publication title is missing",
                        severity=ValidationSeverity.WARNING,
                        issue_type="missing_optional_field",
                        current_value=None,
                        expected_value="publication title"
                    ))
                
                if not pub_meta.get("authors"):
                    issues.append(QualityIssue(
                        field="publication_metadata.authors",
                        message="Publication authors are missing",
                        severity=ValidationSeverity.WARNING,
                        issue_type="missing_optional_field",
                        current_value=None,
                        expected_value="list of authors"
                    ))
        
        # Check dataset analysis completeness
        if "dataset_analysis" in analysis_data:
            dataset_analysis = analysis_data["dataset_analysis"]
            if isinstance(dataset_analysis, dict):
                validated_datasets = dataset_analysis.get("validated_datasets", [])
                newly_discovered = dataset_analysis.get("newly_discovered_datasets", [])
                
                if not validated_datasets and not newly_discovered:
                    issues.append(QualityIssue(
                        field="dataset_analysis",
                        message="No datasets found in analysis",
                        severity=ValidationSeverity.INFO,
                        issue_type="no_datasets_found",
                        current_value=0,
                        expected_value="at least one dataset"
                    ))
        
        # Check code extraction completeness
        if "code_extraction" in analysis_data:
            code_extraction = analysis_data["code_extraction"]
            if isinstance(code_extraction, dict):
                code_snippets = code_extraction.get("extracted_code_snippets", [])
                
                if not code_snippets:
                    issues.append(QualityIssue(
                        field="code_extraction",
                        message="No code snippets extracted",
                        severity=ValidationSeverity.INFO,
                        issue_type="no_code_snippets",
                        current_value=0,
                        expected_value="code snippets if present in publication"
                    ))
        
        return issues
    
    async def _validate_data_consistency(self, analysis_data: Dict[str, Any]) -> List[QualityIssue]:
        """Validate data consistency across different sections."""
        issues = []
        
        # Check workflow status consistency
        workflow_status = analysis_data.get("workflow_status")
        if workflow_status == "completed":
            # If workflow is completed, should have final results
            if not analysis_data.get("final_json"):
                issues.append(QualityIssue(
                    field="workflow_status",
                    message="Workflow marked as completed but no final JSON present",
                    severity=ValidationSeverity.ERROR,
                    issue_type="inconsistent_workflow_status",
                    current_value=workflow_status,
                    expected_value="final_json present for completed workflow"
                ))
        
        # Check dataset counts consistency
        if "dataset_analysis" in analysis_data and "analysis_flags" in analysis_data:
            dataset_analysis = analysis_data["dataset_analysis"]
            analysis_flags = analysis_data["analysis_flags"]
            
            if isinstance(dataset_analysis, dict) and isinstance(analysis_flags, dict):
                total_datasets = len(dataset_analysis.get("validated_datasets", [])) + \
                               len(dataset_analysis.get("newly_discovered_datasets", []))
                
                classification = analysis_flags.get("analysis_classification", {})
                if isinstance(classification, dict):
                    reported_count = classification.get("dataset_count", 0)
                    
                    if total_datasets != reported_count:
                        issues.append(QualityIssue(
                            field="dataset_analysis",
                            message=f"Dataset count mismatch: {total_datasets} actual vs {reported_count} reported",
                            severity=ValidationSeverity.WARNING,
                            issue_type="dataset_count_mismatch",
                            current_value=total_datasets,
                            expected_value=reported_count
                        ))
        
        # Check code snippet counts consistency
        if "code_extraction" in analysis_data and "analysis_flags" in analysis_data:
            code_extraction = analysis_data["code_extraction"]
            analysis_flags = analysis_data["analysis_flags"]
            
            if isinstance(code_extraction, dict) and isinstance(analysis_flags, dict):
                actual_code_count = len(code_extraction.get("extracted_code_snippets", []))
                
                classification = analysis_flags.get("analysis_classification", {})
                if isinstance(classification, dict):
                    reported_count = classification.get("code_snippets_count", 0)
                    
                    if actual_code_count != reported_count:
                        issues.append(QualityIssue(
                            field="code_extraction",
                            message=f"Code snippet count mismatch: {actual_code_count} actual vs {reported_count} reported",
                            severity=ValidationSeverity.WARNING,
                            issue_type="code_count_mismatch",
                            current_value=actual_code_count,
                            expected_value=reported_count
                        ))
        
        return issues
    
    async def _validate_logical_rules(self, analysis_data: Dict[str, Any]) -> List[QualityIssue]:
        """Validate logical rules and business logic."""
        issues = []
        
        # Check that if datasets are found, has_datasets flag should be True
        if "dataset_analysis" in analysis_data and "analysis_flags" in analysis_data:
            dataset_analysis = analysis_data["dataset_analysis"]
            analysis_flags = analysis_data["analysis_flags"]
            
            if isinstance(dataset_analysis, dict) and isinstance(analysis_flags, dict):
                total_datasets = len(dataset_analysis.get("validated_datasets", [])) + \
                               len(dataset_analysis.get("newly_discovered_datasets", []))
                
                has_datasets_flag = analysis_flags.get("has_datasets")
                
                if total_datasets > 0 and has_datasets_flag is False:
                    issues.append(QualityIssue(
                        field="analysis_flags.has_datasets",
                        message="Datasets found but has_datasets flag is False",
                        severity=ValidationSeverity.ERROR,
                        issue_type="logical_inconsistency",
                        current_value=has_datasets_flag,
                        expected_value=True
                    ))
                elif total_datasets == 0 and has_datasets_flag is True:
                    issues.append(QualityIssue(
                        field="analysis_flags.has_datasets",
                        message="No datasets found but has_datasets flag is True",
                        severity=ValidationSeverity.ERROR,
                        issue_type="logical_inconsistency",
                        current_value=has_datasets_flag,
                        expected_value=False
                    ))
        
        # Check that if code snippets are found, is_data_analysis should be True
        if "code_extraction" in analysis_data and "analysis_flags" in analysis_data:
            code_extraction = analysis_data["code_extraction"]
            analysis_flags = analysis_data["analysis_flags"]
            
            if isinstance(code_extraction, dict) and isinstance(analysis_flags, dict):
                code_snippets = code_extraction.get("extracted_code_snippets", [])
                
                is_data_analysis_flag = analysis_flags.get("is_data_analysis")
                
                if len(code_snippets) > 0 and is_data_analysis_flag is False:
                    issues.append(QualityIssue(
                        field="analysis_flags.is_data_analysis",
                        message="Code snippets found but is_data_analysis flag is False",
                        severity=ValidationSeverity.WARNING,
                        issue_type="logical_inconsistency",
                        current_value=is_data_analysis_flag,
                        expected_value=True
                    ))
        
        # Check workflow completion percentage consistency
        if "workflow_metadata" in analysis_data:
            workflow_meta = analysis_data["workflow_metadata"]
            if isinstance(workflow_meta, dict):
                step_count = workflow_meta.get("step_count", 0)
                total_steps = workflow_meta.get("total_steps", 7)
                completion_percentage = workflow_meta.get("completion_percentage", 0)
                
                expected_percentage = (step_count / total_steps) * 100 if total_steps > 0 else 0
                
                if abs(completion_percentage - expected_percentage) > 5.0:  # Allow 5% tolerance
                    issues.append(QualityIssue(
                        field="workflow_metadata.completion_percentage",
                        message=f"Completion percentage inconsistent: {completion_percentage}% vs expected {expected_percentage:.1f}%",
                        severity=ValidationSeverity.WARNING,
                        issue_type="completion_percentage_mismatch",
                        current_value=completion_percentage,
                        expected_value=expected_percentage
                    ))
        
        return issues
    
    async def _validate_dataset_references(self, analysis_data: Dict[str, Any]) -> List[QualityIssue]:
        """Validate dataset references against known datasets."""
        issues = []
        
        if not self.dataset_service:
            logger.warning("DatasetService not available for dataset reference validation")
            return issues
        
        if "dataset_analysis" in analysis_data:
            dataset_analysis = analysis_data["dataset_analysis"]
            if isinstance(dataset_analysis, dict):
                validated_datasets = dataset_analysis.get("validated_datasets", [])
                
                for dataset in validated_datasets:
                    if isinstance(dataset, dict):
                        dataset_name = dataset.get("name")
                        if dataset_name:
                            try:
                                # Check if dataset exists in database
                                existing_datasets = await self.dataset_service.get_datasets_by_aliases([dataset_name])
                                if not existing_datasets:
                                    issues.append(QualityIssue(
                                        field=f"dataset_analysis.validated_datasets.{dataset_name}",
                                        message=f"Dataset '{dataset_name}' not found in database",
                                        severity=ValidationSeverity.WARNING,
                                        issue_type="unknown_dataset",
                                        current_value=dataset_name,
                                        expected_value="known dataset in database"
                                    ))
                            except Exception as e:
                                logger.warning(f"Error validating dataset {dataset_name}: {e}")
        
        return issues
    
    async def _validate_code_quality(self, analysis_data: Dict[str, Any]) -> List[QualityIssue]:
        """Validate quality of extracted code snippets."""
        issues = []
        
        if "code_extraction" in analysis_data:
            code_extraction = analysis_data["code_extraction"]
            if isinstance(code_extraction, dict):
                code_snippets = code_extraction.get("extracted_code_snippets", [])
                
                for i, snippet in enumerate(code_snippets):
                    if isinstance(snippet, dict):
                        # Check for empty code content
                        content = snippet.get("content", "")
                        if not content or len(content.strip()) < 10:
                            issues.append(QualityIssue(
                                field=f"code_extraction.extracted_code_snippets[{i}].content",
                                message="Code snippet content is too short or empty",
                                severity=ValidationSeverity.WARNING,
                                issue_type="short_code_snippet",
                                current_value=len(content),
                                expected_value="at least 10 characters"
                            ))
                        
                        # Check for missing language
                        language = snippet.get("language", "")
                        if not language:
                            issues.append(QualityIssue(
                                field=f"code_extraction.extracted_code_snippets[{i}].language",
                                message="Programming language not specified",
                                severity=ValidationSeverity.WARNING,
                                issue_type="missing_language",
                                current_value=language,
                                expected_value="programming language identifier"
                            ))
                        
                        # Check relevance score range
                        relevance_score = snippet.get("relevance_score", 0)
                        if not (0 <= relevance_score <= 10):
                            issues.append(QualityIssue(
                                field=f"code_extraction.extracted_code_snippets[{i}].relevance_score",
                                message="Relevance score out of valid range (0-10)",
                                severity=ValidationSeverity.ERROR,
                                issue_type="invalid_relevance_score",
                                current_value=relevance_score,
                                expected_value="value between 0 and 10"
                            ))
        
        return issues
    
    async def _validate_link_quality(self, analysis_data: Dict[str, Any]) -> List[QualityIssue]:
        """Validate quality of extracted links."""
        issues = []
        
        if "link_extraction" in analysis_data:
            link_extraction = analysis_data["link_extraction"]
            if isinstance(link_extraction, dict):
                external_links = link_extraction.get("external_links", [])
                github_repos = link_extraction.get("github_repositories", [])
                
                # Validate external links
                for i, link in enumerate(external_links):
                    if isinstance(link, dict):
                        url = link.get("url", "")
                        if not url or not url.startswith(("http://", "https://")):
                            issues.append(QualityIssue(
                                field=f"link_extraction.external_links[{i}].url",
                                message="Invalid URL format",
                                severity=ValidationSeverity.ERROR,
                                issue_type="invalid_url",
                                current_value=url,
                                expected_value="valid HTTP/HTTPS URL"
                            ))
                        
                        # Check relevance score
                        relevance_score = link.get("relevance_score", 0)
                        if not (0 <= relevance_score <= 10):
                            issues.append(QualityIssue(
                                field=f"link_extraction.external_links[{i}].relevance_score",
                                message="Relevance score out of valid range (0-10)",
                                severity=ValidationSeverity.ERROR,
                                issue_type="invalid_relevance_score",
                                current_value=relevance_score,
                                expected_value="value between 0 and 10"
                            ))
                
                # Validate GitHub repositories
                for i, repo in enumerate(github_repos):
                    if isinstance(repo, dict):
                        url = repo.get("url", "")
                        if not url or "github.com" not in url:
                            issues.append(QualityIssue(
                                field=f"link_extraction.github_repositories[{i}].url",
                                message="Invalid GitHub repository URL",
                                severity=ValidationSeverity.ERROR,
                                issue_type="invalid_github_url",
                                current_value=url,
                                expected_value="valid GitHub repository URL"
                            ))
                        
                        # Check required GitHub fields
                        owner = repo.get("owner", "")
                        repository = repo.get("repository", "")
                        if not owner or not repository:
                            issues.append(QualityIssue(
                                field=f"link_extraction.github_repositories[{i}]",
                                message="Missing owner or repository name",
                                severity=ValidationSeverity.WARNING,
                                issue_type="missing_github_fields",
                                current_value={"owner": owner, "repository": repository},
                                expected_value="both owner and repository names"
                            ))
        
        return issues
    
    def _calculate_completeness_score(self, analysis_data: Dict[str, Any], issues: List[QualityIssue]) -> CompletenessScore:
        """Calculate completeness score for analysis data."""
        # Define field weights for completeness scoring
        field_weights = {
            "publication_metadata": 0.25,
            "dataset_analysis": 0.25,
            "code_extraction": 0.20,
            "link_extraction": 0.15,
            "workflow_metadata": 0.15
        }
        
        scores = {}
        missing_critical = []
        missing_optional = []
        
        # Calculate scores for each section
        for section, weight in field_weights.items():
            if section in analysis_data and analysis_data[section]:
                section_data = analysis_data[section]
                if isinstance(section_data, dict):
                    # Count filled vs empty fields
                    total_fields = len(section_data)
                    filled_fields = sum(1 for v in section_data.values() if v is not None and v != "")
                    scores[section] = (filled_fields / total_fields) * 100 if total_fields > 0 else 100
                else:
                    scores[section] = 100  # Non-dict sections are considered complete
            else:
                scores[section] = 0
                missing_critical.append(section)
        
        # Calculate overall score
        overall_score = sum(scores.get(section, 0) * weight for section, weight in field_weights.items())
        
        # Identify missing optional fields from issues
        for issue in issues:
            if issue.issue_type == "missing_optional_field":
                missing_optional.append(issue.field)
        
        return CompletenessScore(
            overall_score=overall_score,
            publication_metadata_score=scores.get("publication_metadata", 0),
            dataset_analysis_score=scores.get("dataset_analysis", 0),
            code_extraction_score=scores.get("code_extraction", 0),
            link_extraction_score=scores.get("link_extraction", 0),
            workflow_metadata_score=scores.get("workflow_metadata", 0),
            missing_critical_fields=missing_critical,
            missing_optional_fields=missing_optional
        )
    
    def _calculate_consistency_score(self, issues: List[QualityIssue]) -> float:
        """Calculate data consistency score based on consistency issues."""
        consistency_issues = [i for i in issues if "consistency" in i.issue_type or "mismatch" in i.issue_type]
        
        if not consistency_issues:
            return 100.0
        
        # Penalize based on severity
        total_penalty = 0
        for issue in consistency_issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                total_penalty += 20
            elif issue.severity == ValidationSeverity.ERROR:
                total_penalty += 10
            elif issue.severity == ValidationSeverity.WARNING:
                total_penalty += 5
            else:
                total_penalty += 1
        
        return max(0, 100 - total_penalty)
    
    def _calculate_logical_score(self, issues: List[QualityIssue]) -> float:
        """Calculate logical validation score based on logical issues."""
        logical_issues = [i for i in issues if "logical" in i.issue_type or "inconsistency" in i.issue_type]
        
        if not logical_issues:
            return 100.0
        
        # Penalize based on severity
        total_penalty = 0
        for issue in logical_issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                total_penalty += 25
            elif issue.severity == ValidationSeverity.ERROR:
                total_penalty += 15
            elif issue.severity == ValidationSeverity.WARNING:
                total_penalty += 8
            else:
                total_penalty += 2
        
        return max(0, 100 - total_penalty)
    
    def _determine_quality_level(self, completeness_score: CompletenessScore, issues: List[QualityIssue]) -> QualityLevel:
        """Determine overall quality level based on scores and issues."""
        # Count issues by severity
        critical_issues = len([i for i in issues if i.severity == ValidationSeverity.CRITICAL])
        error_issues = len([i for i in issues if i.severity == ValidationSeverity.ERROR])
        warning_issues = len([i for i in issues if i.severity == ValidationSeverity.WARNING])
        
        # Determine quality level based on completeness and issues
        if critical_issues > 0:
            return QualityLevel.CRITICAL
        elif error_issues > 0:
            return QualityLevel.POOR
        elif warning_issues > 0:
            # If there are warnings, quality should be GOOD or FAIR depending on completeness
            if completeness_score.overall_score >= 75:
                return QualityLevel.GOOD
            else:
                return QualityLevel.FAIR
        elif completeness_score.overall_score >= 90:
            return QualityLevel.EXCELLENT
        elif completeness_score.overall_score >= 75:
            return QualityLevel.GOOD
        elif completeness_score.overall_score >= 50:
            return QualityLevel.FAIR
        else:
            return QualityLevel.POOR
    
    def get_quality_report(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """
        Generate a comprehensive quality report.
        
        Args:
            metrics: QualityMetrics from validation
            
        Returns:
            Detailed quality report
        """
        return {
            "overall_quality": {
                "level": metrics.overall_quality_level.value,
                "completeness_score": metrics.completeness_score.overall_score,
                "consistency_score": metrics.data_consistency_score,
                "logical_score": metrics.logical_validation_score
            },
            "issue_summary": {
                "total_issues": metrics.total_issues,
                "critical_issues": metrics.critical_issues,
                "warnings": metrics.warnings,
                "info_issues": metrics.info_issues
            },
            "completeness_details": {
                "publication_metadata": metrics.completeness_score.publication_metadata_score,
                "dataset_analysis": metrics.completeness_score.dataset_analysis_score,
                "code_extraction": metrics.completeness_score.code_extraction_score,
                "link_extraction": metrics.completeness_score.link_extraction_score,
                "workflow_metadata": metrics.completeness_score.workflow_metadata_score,
                "missing_critical_fields": metrics.completeness_score.missing_critical_fields,
                "missing_optional_fields": metrics.completeness_score.missing_optional_fields
            },
            "quality_issues": [
                {
                    "field": issue.field,
                    "message": issue.message,
                    "severity": issue.severity.value,
                    "type": issue.issue_type,
                    "current_value": issue.current_value,
                    "expected_value": issue.expected_value,
                    "suggestion": issue.suggestion
                }
                for issue in metrics.quality_issues
            ],
            "validation_timestamp": metrics.validation_timestamp.isoformat()
        } 