"""
Code and Link Extraction Agent for Publication Analysis.

This module implements the CodeExtractionAgent that extracts GitHub links,
code snippets, and external resources from publication text using pattern
matching and LLM analysis.
"""

import logging
import re
import urllib.parse
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import httpx

from ..services.llm_service import LLMService, PromptTemplate
from ..services.code_analysis_llm import CodeAnalysisLLM, CodeAnalysisConfig, CodeAnalysisResult
from ..services.link_validator import LinkValidator, LinkValidationConfig, LinkValidationResult
from ..workflows.state_models import AnalysisState

logger = logging.getLogger(__name__)


class LinkType(Enum):
    """Types of links that can be extracted."""
    GITHUB = "github"
    DATA_REPOSITORY = "data_repository"
    DOCUMENTATION = "documentation"
    EXTERNAL_TOOL = "external_tool"
    ACADEMIC_RESOURCE = "academic_resource"
    OTHER = "other"


class CodeType(Enum):
    """Types of code that can be extracted."""
    PYTHON = "python"
    R = "r"
    JAVASCRIPT = "javascript"
    SQL = "sql"
    BASH = "bash"
    YAML = "yaml"
    JSON = "json"
    MARKDOWN = "markdown"
    OTHER = "other"


@dataclass
class ExtractionConfig:
    """Configuration for the CodeExtractionAgent."""
    max_text_length: int = 400000  # Maximum text length to process
    temperature: float = 0.7  # LLM temperature for code analysis
    max_tokens: int = 4000  # Response length for analysis
    link_validation_timeout: int = 5  # Timeout for link validation (seconds)
    min_code_snippet_length: int = 10  # Minimum length for code snippets
    max_code_snippet_length: int = 2000  # Maximum length for code snippets
    relevance_threshold: float = 6.0  # Minimum relevance score (0-10)
    validate_links: bool = True  # Whether to validate links by making HTTP requests


@dataclass
class GitHubInfo:
    """Information extracted from GitHub URLs."""
    url: str
    owner: str
    repository: str
    path: Optional[str] = None
    branch: Optional[str] = None
    is_valid: Optional[bool] = None
    description: Optional[str] = None
    language: Optional[str] = None
    stars: Optional[int] = None
    topics: Optional[List[str]] = None


@dataclass
class CodeSnippet:
    """Information about extracted code snippets."""
    content: str
    language: CodeType
    context: str
    start_position: int
    end_position: int
    relevance_score: float
    description: Optional[str] = None
    purpose: Optional[str] = None  # e.g., "data processing", "visualization", "modeling"


@dataclass
class ExternalLink:
    """Information about external links."""
    url: str
    link_type: LinkType
    title: Optional[str] = None
    description: Optional[str] = None
    context: str = ""
    is_accessible: Optional[bool] = None
    relevance_score: float = 0.0


@dataclass
class ExtractionResult:
    """Result from code and link extraction."""
    github_repositories: List[GitHubInfo]
    code_snippets: List[CodeSnippet]
    external_links: List[ExternalLink]
    programming_languages: Set[str]
    total_code_blocks: int
    total_links_found: int
    processing_time: float
    errors: List[str] = field(default_factory=list)


class CodeExtractionAgent:
    """
    Agent to extract GitHub links, code snippets, and external resources from publications.
    
    Uses regex patterns for initial extraction and LLM analysis for categorization,
    relevance scoring, and context understanding.
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        config: Optional[ExtractionConfig] = None,
        code_analysis_config: Optional[CodeAnalysisConfig] = None,
        link_validation_config: Optional[LinkValidationConfig] = None,
        github_token: Optional[str] = None
    ) -> None:
        """
        Initialize the CodeExtractionAgent.
        
        Args:
            llm_service: LLM service for code analysis
            config: Configuration for extraction
            code_analysis_config: Configuration for code analysis module
            link_validation_config: Configuration for link validation
            github_token: Optional GitHub API token for enhanced metadata
        """
        self.llm_service = llm_service
        self.config = config or ExtractionConfig()
        
        # Initialize the dedicated code analysis module
        self.code_analyzer = CodeAnalysisLLM(
            llm_service=llm_service,
            config=code_analysis_config or CodeAnalysisConfig(
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                relevance_threshold=self.config.relevance_threshold
            )
        )
        
        # Initialize the dedicated link validation system
        self.link_validator = LinkValidator(
            config=link_validation_config or LinkValidationConfig(
                timeout=self.config.link_validation_timeout
            ),
            github_token=github_token
        )
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Initialize prompt templates for LLM discovery and link analysis
        self._setup_prompt_templates()
        
        logger.info("CodeExtractionAgent initialized with code analysis and link validation modules")
    
    async def close(self) -> None:
        """Close the agent and cleanup resources."""
        try:
            await self.link_validator.close()
            logger.info("CodeExtractionAgent resources cleaned up")
        except Exception as e:
            logger.error(f"Error closing CodeExtractionAgent: {e}")
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for code and link extraction."""
        
        # GitHub URL patterns
        self.github_patterns = [
            # Standard GitHub URLs
            re.compile(
                r'https?://(?:www\.)?github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+)'
                r'(?=[\s.,;!?]|$|/)',
                re.IGNORECASE
            ),
            # GitHub raw content URLs
            re.compile(
                r'https?://raw\.githubusercontent\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+)'
                r'/([a-zA-Z0-9._/-]+)/(.+)',
                re.IGNORECASE
            ),
            # GitHub gist URLs
            re.compile(
                r'https?://gist\.github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9]+)',
                re.IGNORECASE
            )
        ]
        
        # Code block patterns
        self.code_block_patterns = [
            # Markdown code blocks with language
            re.compile(r'```(\w+)?\s*\n(.*?)\n\s*```', re.DOTALL | re.MULTILINE),
            # Indented code blocks (4+ spaces)
            re.compile(r'^(?: {4,}|\t+)(.+)$', re.MULTILINE),
            # Inline code
            re.compile(r'`([^`\n]+)`'),
            # LaTeX code environments
            re.compile(r'\\begin\{(?:lstlisting|verbatim|code)\}(.*?)\\end\{(?:lstlisting|verbatim|code)\}', re.DOTALL)
        ]
        
        # External link patterns
        self.link_patterns = [
            # General HTTP/HTTPS URLs
            re.compile(
                r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?',
                re.IGNORECASE
            ),
            # DOI links
            re.compile(r'(?:https?://)?(?:dx\.)?doi\.org/([^\s]+)', re.IGNORECASE),
            # arXiv links
            re.compile(r'(?:https?://)?arxiv\.org/(?:abs|pdf)/([^\s]+)', re.IGNORECASE)
        ]
        
        # Programming language detection patterns
        self.language_patterns = {
            CodeType.PYTHON: [
                re.compile(r'\bimport\s+\w+', re.IGNORECASE),
                re.compile(r'\bfrom\s+\w+\s+import', re.IGNORECASE),
                re.compile(r'\bdef\s+\w+\s*\(', re.IGNORECASE),
                re.compile(r'\bclass\s+\w+\s*\(', re.IGNORECASE),
                re.compile(r'print\s*\(', re.IGNORECASE),
            ],
            CodeType.R: [
                re.compile(r'\blibrary\s*\(', re.IGNORECASE),
                re.compile(r'\brequire\s*\(', re.IGNORECASE),
                re.compile(r'<-', re.IGNORECASE),
                re.compile(r'\bdata\.frame\s*\(', re.IGNORECASE),
                re.compile(r'\bggplot\s*\(', re.IGNORECASE),
            ],
            CodeType.SQL: [
                re.compile(r'\bSELECT\s+', re.IGNORECASE),
                re.compile(r'\bFROM\s+', re.IGNORECASE),
                re.compile(r'\bWHERE\s+', re.IGNORECASE),
                re.compile(r'\bJOIN\s+', re.IGNORECASE),
            ],
            CodeType.BASH: [
                re.compile(r'#!/bin/(?:bash|sh)', re.IGNORECASE),
                re.compile(r'\b(?:ls|cd|mkdir|rm|cp|mv|grep|awk|sed)\s+', re.IGNORECASE),
                re.compile(r'\$\{?\w+\}?'),
            ],
            CodeType.JAVASCRIPT: [
                re.compile(r'\bfunction\s+\w+\s*\(', re.IGNORECASE),
                re.compile(r'\bconst\s+\w+\s*=', re.IGNORECASE),
                re.compile(r'\blet\s+\w+\s*=', re.IGNORECASE),
                re.compile(r'console\.log\s*\(', re.IGNORECASE),
            ]
        }
        
        logger.info("Regex patterns compiled for code and link extraction")
    
    def _setup_prompt_templates(self) -> None:
        """Setup prompt templates for LLM-based discovery, validation, and link analysis."""
        
        # Comprehensive content discovery prompt
        content_discovery_prompt = PromptTemplate(
            name="content_discovery",
            template="""You are an expert at analyzing research papers to find code snippets, GitHub repositories, and external links. Your task is to comprehensively analyze the publication text and discover ALL code-related content and links.

PUBLICATION TEXT:
{publication_text}

REGEX FINDINGS (for validation):
GitHub Repositories Found: {found_github_repos}
External Links Found: {found_external_links}

TASKS:
1. **VALIDATE REGEX FINDINGS**: Review the regex-detected items above. Are they correct? Are any false positives?

2. **COMPREHENSIVE LINK DISCOVERY**: Thoroughly analyze the full text to find ALL links and repositories that regex patterns might have missed:
   
   **GitHub Repositories:**
   - Full URLs: github.com/user/repo, https://github.com/
   - Partial mentions: "user/repo", "github.com/user/repo", "user-name/repo-name"
   - Repository references: "the repository at user/repo", "available on GitHub as user/repo"
   - Code hosting platforms: GitLab, Bitbucket, SourceForge, etc.
   
   **External Links & Data Sources:**
   - Data repositories: Zenodo, Figshare, Dryad, Harvard Dataverse, ICPSR
   - Software tools: CRAN, PyPI, Bioconductor, npm, Maven Central
   - Documentation: ReadTheDocs, Wiki pages, API documentation
   - Supplementary materials: OSF, GitHub releases, project websites
   - Research platforms: Kaggle, UCI ML Repository, OpenML
   - Version control: GitLab, Bitbucket, Gitea, Codeberg
   
   **Code References:**
   - Algorithm implementations mentioned in text
   - Software packages and libraries
   - Code snippets in supplementary materials
   - Scripts and tools referenced

3. **EXTRACT RICH CONTEXT**: For each discovered item, provide:
   - Exact mention text from the paper
   - Surrounding context (2-3 sentences)
   - Confidence level based on clarity of reference
   - Type/category of the link/repository

Format your response as JSON:
{{
  "validation": {{
    "github_repos": [{{"url": "<url>", "valid": true/false, "reason": "<reason>"}}],
    "external_links": [{{"url": "<url>", "valid": true/false, "reason": "<reason>"}}]
  }},
  "additional_discoveries": {{
    "github_repos": [{{"url": "<reconstructed_url>", "mention": "<original_text>", "context": "<surrounding_text>", "confidence": <0.0-1.0>}}],
    "external_links": [{{"url": "<url>", "mention": "<original_text>", "context": "<surrounding_text>", "type": "<type>", "confidence": <0.0-1.0>}}]
  }}
}}

IMPORTANT: Be thorough and systematic. Look for any mention of code, software, data, or links, even if they appear in footnotes, references, or supplementary material sections. Prioritize finding actual usable links and repositories over general mentions.
""",
            variables=["publication_text", "found_github_repos", "found_external_links"]
        )
        
        # Link analysis prompt (unchanged)
        link_analysis_prompt = PromptTemplate(
            name="link_analysis",
            template="""You are an expert at analyzing external links in research papers. Your task is to categorize links and assess their relevance to the research.

LINKS TO ANALYZE:
{links}

PAPER CONTEXT:
{paper_context}

LINK CATEGORIES:
- github: GitHub repositories or code
- data_repository: Data repositories (Zenodo, Figshare, etc.)
- documentation: Documentation, tutorials, guides
- external_tool: Software tools, libraries, APIs
- academic_resource: Papers, journals, academic databases
- other: Other types of resources

For each link, provide:
1. Category from the list above
2. Relevance score from 0-10 (10 = highly relevant, 0 = not relevant)
3. Brief description of what the link provides
4. Title if available from context

Format your response as JSON array:
[
  {{
    "url": "<url>",
    "category": "<category>",
    "relevance_score": <0-10>,
    "description": "<brief description>",
    "title": "<title if available>"
  }}
]""",
            variables=["links", "paper_context"]
        )
        
        self.llm_service.add_prompt_template(content_discovery_prompt)
        self.llm_service.add_prompt_template(link_analysis_prompt)
        
        logger.info("Content discovery, validation, and link analysis prompt templates configured")
    
    async def extract_code_and_links(self, state: AnalysisState) -> ExtractionResult:
        """
        Extract code snippets and links from the publication.
        
        Args:
            state: Current analysis state with publication content
            
        Returns:
            ExtractionResult with extracted code and links
        """
        import time
        start_time = time.time()

        if not state.is_data_analysis:
            return ExtractionResult(
                github_repositories=[],
                code_snippets=[],
                external_links=[],
                programming_languages=set(),
                total_code_blocks=0,
                total_links_found=0,
                processing_time=time.time() - start_time,
                errors=["Not a data analysis publication"]
            )
        
        try:
            logger.info(f"Starting code and link extraction for publication: {state.publication_id}")
            
            # Extract and preprocess text
            text_content = state.raw_text
            if not text_content:
                logger.warning("No text content available for extraction")
                return ExtractionResult(
                    github_repositories=[],
                    code_snippets=[],
                    external_links=[],
                    programming_languages=set(),
                    total_code_blocks=0,
                    total_links_found=0,
                    processing_time=time.time() - start_time,
                    errors=["No text content available"]
                )
            
            # Phase 1: Regex-based extraction
            logger.info("Phase 1: Regex-based content extraction")
            
            # Extract GitHub repositories
            github_repos = self._extract_github_repositories(text_content)
            
            # Extract external links
            external_links = self._extract_external_links(text_content)

            logger.info(f"Phase 1 results: {len(github_repos)} GitHub repos, {len(external_links)} external links")
            
            # Phase 2: LLM-based discovery and validation
            logger.info("Phase 2: LLM-based content discovery and validation")
            llm_discovery_result = await self._llm_content_discovery_and_validation(
                text_content, github_repos, external_links
            )

            # Combine regex and LLM findings
            github_repos = self._combine_github_repos(github_repos, llm_discovery_result)
            external_links = self._combine_external_links(external_links, llm_discovery_result)

            if external_links:
                analyzed_links = await self._analyze_external_links(external_links, text_content)
                # Enhance with validation metadata if validation is enabled
                if self.config.validate_links:
                    validated_links = await self._validate_external_links_enhanced(analyzed_links)
                    external_links = validated_links
                else:
                    external_links = analyzed_links
            # Validate GitHub repositories with comprehensive link validation
            if github_repos and self.config.validate_links:
                validated_repos = await self._validate_github_repositories_enhanced(github_repos)
                github_repos = validated_repos
            
            relevant_links = [
                link for link in external_links
            ]
            
            # Extract programming languages
            processing_time = time.time() - start_time
            logger.info(
                f"Enhanced code and link extraction completed - Found {len(github_repos)} GitHub repos, "
                f"{len(relevant_links)} external links "
                f"using combined regex + LLM approach in {processing_time:.2f}s"
            )
            
            return ExtractionResult(
                github_repositories=github_repos,
                code_snippets=[],
                external_links=relevant_links,
                programming_languages=[],
                total_code_blocks=0,
                total_links_found=len(external_links) + len(github_repos),
                processing_time=processing_time,
                errors=[]
            )
            
        except Exception as e:
            logger.error(f"Error in code and link extraction: {e}")
            return ExtractionResult(
                github_repositories=[],
                code_snippets=[],
                external_links=[],
                programming_languages=set(),
                total_code_blocks=0,
                total_links_found=0,
                processing_time=time.time() - start_time,
                errors=[str(e)]
            )
    
    def _extract_text_content(self, state: AnalysisState) -> str:
        """
        Extract relevant text content from the publication.
        
        Args:
            state: Analysis state with GROBID content
            
        Returns:
            Preprocessed text content for analysis
        """
        if not state.grobid_content:
            return state.raw_text or ""
        
        content_parts = []
        grobid_data = state.grobid_content
        
        # Extract title
        if "title" in grobid_data:
            content_parts.append(f"TITLE: {grobid_data['title']}")
        
        # Extract abstract
        if "abstract" in grobid_data:
            content_parts.append(f"ABSTRACT: {grobid_data['abstract']}")
        
        # Extract all sections (prioritize methods, implementation, results)
        if "sections" in grobid_data and isinstance(grobid_data["sections"], list):
            for section in grobid_data["sections"]:
                if isinstance(section, dict) and section.get("text"):
                    section_title = section.get("title", "").upper()
                    content_parts.append(f"{section_title}: {section['text']}")
        
        # Combine and preprocess
        full_text = "\n\n".join(content_parts)
        return self._preprocess_text(full_text)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for code and link extraction.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned and truncated text
        """
        if not text:
            return ""
        
        # Clean up whitespace but preserve code formatting
        text = re.sub(r'\n{3,}', '\n\n', text)  # Reduce excessive line breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces but keep structure
        
        # Truncate to maximum length
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length] + "..."
            logger.info(f"Text truncated to {self.config.max_text_length} characters")
        
        return text
    
    def _extract_github_repositories(self, text: str) -> List[GitHubInfo]:
        """
        Extract GitHub repository information from text.
        
        Args:
            text: Text content to search
            
        Returns:
            List of GitHubInfo objects
        """
        github_repos = []
        seen_repos = set()
        
        for pattern in self.github_patterns:
            for match in pattern.finditer(text):
                if pattern == self.github_patterns[0]:  # Standard GitHub URLs
                    owner, repo = match.groups()
                    repo_key = f"{owner}/{repo}"
                    
                    if repo_key not in seen_repos:
                        seen_repos.add(repo_key)
                        # Remove trailing punctuation from repository name and URL
                        clean_repo = repo.rstrip('.,;!?')
                        clean_url = match.group(0).rstrip('.,;!?')
                        github_info = GitHubInfo(
                            url=clean_url,
                            owner=owner,
                            repository=clean_repo,
                            path=None,
                            branch=None
                        )
                        github_repos.append(github_info)
                
                elif pattern == self.github_patterns[1]:  # Raw content URLs
                    owner, repo, branch, path = match.groups()
                    repo_key = f"{owner}/{repo}"
                    
                    if repo_key not in seen_repos:
                        seen_repos.add(repo_key)
                        github_info = GitHubInfo(
                            url=f"https://github.com/{owner}/{repo}",
                            owner=owner,
                            repository=repo,
                            path=path,
                            branch=branch
                        )
                        github_repos.append(github_info)
                
                elif pattern == self.github_patterns[2]:  # Gist URLs
                    owner, gist_id = match.groups()
                    repo_key = f"{owner}/gist-{gist_id}"
                    
                    if repo_key not in seen_repos:
                        seen_repos.add(repo_key)
                        github_info = GitHubInfo(
                            url=match.group(0),
                            owner=owner,
                            repository=f"gist-{gist_id}",
                            path=None,
                            branch=None
                        )
                        github_repos.append(github_info)
        
        logger.info(f"Extracted {len(github_repos)} GitHub repositories")
        return github_repos
    
    def _extract_code_snippets(self, text: str) -> List[CodeSnippet]:
        """
        Extract code snippets from text.
        
        Args:
            text: Text content to search
            
        Returns:
            List of CodeSnippet objects
        """
        code_snippets = []
        
        for pattern_idx, pattern in enumerate(self.code_block_patterns):
            for match in pattern.finditer(text):
                if pattern_idx == 0:  # Markdown code blocks
                    language_hint = match.group(1)
                    code_content = match.group(2)
                elif pattern_idx == 1:  # Indented blocks
                    language_hint = None
                    code_content = match.group(1)
                elif pattern_idx == 2:  # Inline code
                    language_hint = None
                    code_content = match.group(1)
                else:  # LaTeX code environments
                    language_hint = None
                    code_content = match.group(1)
                
                # Filter by length
                if (len(code_content.strip()) < self.config.min_code_snippet_length or 
                    len(code_content.strip()) > self.config.max_code_snippet_length):
                    continue
                
                # Detect programming language
                detected_language = self._detect_programming_language(code_content, language_hint)
                
                # Extract context around the code snippet
                start_pos = match.start()
                end_pos = match.end()
                context = self._extract_context_around_position(text, start_pos, end_pos)
                
                snippet = CodeSnippet(
                    content=code_content.strip(),
                    language=detected_language,
                    context=context,
                    start_position=start_pos,
                    end_position=end_pos,
                    relevance_score=0.0  # Will be set by LLM analysis
                )
                code_snippets.append(snippet)
        
        logger.info(f"Extracted {len(code_snippets)} code snippets")
        return code_snippets
    
    def _extract_external_links(self, text: str) -> List[ExternalLink]:
        """
        Extract external links from text.
        
        Args:
            text: Text content to search
            
        Returns:
            List of ExternalLink objects
        """
        external_links = []
        seen_urls = set()
        
        for pattern in self.link_patterns:
            for match in pattern.finditer(text):
                url = match.group(0)
                
                # Skip GitHub URLs (handled separately)
                if 'github.com' in url.lower() or 'githubusercontent.com' in url.lower():
                    continue
                
                # Skip duplicates
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                
                # Extract context around the link
                start_pos = match.start()
                end_pos = match.end()
                context = self._extract_context_around_position(text, start_pos, end_pos)
                
                # Initial link type classification
                link_type = self._classify_link_type(url)
                
                link = ExternalLink(
                    url=url,
                    link_type=link_type,
                    context=context,
                    relevance_score=0.0  # Will be set by LLM analysis
                )
                external_links.append(link)
        
        logger.info(f"Extracted {len(external_links)} external links")
        return external_links
    
    def _detect_programming_language(self, code: str, language_hint: Optional[str] = None) -> CodeType:
        """
        Detect programming language from code content.
        
        Args:
            code: Code content
            language_hint: Language hint from markup (e.g., from ```python)
            
        Returns:
            Detected CodeType
        """
        # Use language hint if available
        if language_hint:
            hint_lower = language_hint.lower()
            if hint_lower in ['python', 'py']:
                return CodeType.PYTHON
            elif hint_lower in ['r']:
                return CodeType.R
            elif hint_lower in ['javascript', 'js']:
                return CodeType.JAVASCRIPT
            elif hint_lower in ['sql']:
                return CodeType.SQL
            elif hint_lower in ['bash', 'sh', 'shell']:
                return CodeType.BASH
            elif hint_lower in ['yaml', 'yml']:
                return CodeType.YAML
            elif hint_lower in ['json']:
                return CodeType.JSON
        
        # Pattern-based detection
        for lang_type, patterns in self.language_patterns.items():
            match_count = sum(1 for pattern in patterns if pattern.search(code))
            if match_count >= 2:  # Require at least 2 pattern matches
                return lang_type
        
        # Check for specific formats
        if code.strip().startswith('{') and code.strip().endswith('}'):
            try:
                import json
                json.loads(code)
                return CodeType.JSON
            except:
                pass
        
        return CodeType.OTHER
    
    def _classify_link_type(self, url: str) -> LinkType:
        """
        Classify link type based on URL patterns.
        
        Args:
            url: URL to classify
            
        Returns:
            Classified LinkType
        """
        url_lower = url.lower()
        
        # Data repositories
        data_repos = ['zenodo.org', 'figshare.com', 'dataverse.org', 'dryad.org', 
                     'osf.io', 'kaggle.com/datasets']
        if any(repo in url_lower for repo in data_repos):
            return LinkType.DATA_REPOSITORY
        
        # Documentation and tutorials
        docs_sites = ['docs.', 'documentation', 'tutorial', 'readthedocs.io', 
                     'wiki', 'guide']
        if any(doc in url_lower for doc in docs_sites):
            return LinkType.DOCUMENTATION
        
        # Academic resources
        academic_sites = ['doi.org', 'arxiv.org', 'pubmed.ncbi.nlm.nih.gov', 
                         'scholar.google.com', 'researchgate.net']
        if any(academic in url_lower for academic in academic_sites):
            return LinkType.ACADEMIC_RESOURCE
        
        # External tools and APIs
        tool_indicators = ['api', 'tool', 'software', 'app', 'service']
        if any(tool in url_lower for tool in tool_indicators):
            return LinkType.EXTERNAL_TOOL
        
        return LinkType.OTHER
    
    def _extract_context_around_position(self, text: str, start_pos: int, end_pos: int, window: int = 150) -> str:
        """
        Extract context around a specific position in text.
        
        Args:
            text: Full text
            start_pos: Start position of match
            end_pos: End position of match
            window: Characters to include before and after
            
        Returns:
            Context string
        """
        context_start = max(0, start_pos - window)
        context_end = min(len(text), end_pos + window)
        return text[context_start:context_end]
    
    async def _llm_content_discovery_and_validation(
        self,
        text_content: str,
        regex_github_repos: List[GitHubInfo],
        regex_external_links: List[ExternalLink]
    ) -> Dict[str, Any]:
        """
        Use LLM to validate regex findings and discover additional content.
        
        Args:
            text_content: Full publication text
            regex_github_repos: GitHub repos found by regex
            regex_code_snippets: Code snippets found by regex  
            regex_external_links: External links found by regex
            
        Returns:
            Dictionary with validation results and additional discoveries
        """
        try:
            # Prepare regex findings for LLM review
            github_summary = [{"url": repo.url, "owner": repo.owner, "repository": repo.repository} 
                            for repo in regex_github_repos]
            
            link_summary = [{"url": link.url, "type": link.link_type.value} 
                          for link in regex_external_links]
            
            # Get discovery prompt
            prompt_template = self.llm_service.get_prompt_template("content_discovery")
            prompt = prompt_template.render(
                publication_text=text_content,  # Limit text for LLM context
                found_github_repos=str(github_summary),
                found_external_links=str(link_summary)
            )
            # Call LLM for discovery and validation
            response = await self.llm_service.generate(
                prompt=prompt,
                parameters={
                    "temperature": 0.7,  # Low temperature for accuracy
                    "max_tokens": 3000   # Enough for comprehensive response
                }
            )
            
            # Process LLM response
            return self._process_discovery_response(response)
            
        except Exception as e:
            logger.error(f"Error in LLM content discovery: {e}")
            return {"validation": {}, "additional_discoveries": {}}
    
    def _process_discovery_response(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process LLM discovery response.
        
        Args:
            llm_response: Response from LLM service
            
        Returns:
            Parsed discovery results
        """
        try:
            # Extract response text (chat/completions first, fallback to legacy completions)
            response_text = ""
            if "choices" in llm_response and llm_response["choices"]:
                first_choice = llm_response["choices"][0]
                message = first_choice.get("message") or {}
                response_text = (message.get("content") or "").strip()
                if not response_text:
                    response_text = first_choice.get("text", "").strip()
            
            if not response_text:
                logger.warning("Empty response from LLM discovery")
                return {"validation": {}, "additional_discoveries": {}}
            
            # Parse JSON response
            import json
            try:
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if not json_match:
                    logger.warning("No JSON found in LLM response")
                    return {"validation": {}, "additional_discoveries": {}}

                json_data = json_match.group()
                result = json.loads(json_data)
                logger.info(f"LLM discovery found {len(result.get('additional_discoveries', {}).get('github_repos', []))} additional GitHub repos, "
                          f"{len(result.get('additional_discoveries', {}).get('code_snippets', []))} additional code snippets, "
                          f"{len(result.get('additional_discoveries', {}).get('external_links', []))} additional external links")
                return result
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse LLM discovery response as JSON: {e}")
                logger.debug(f"Response text: {response_text}")
                return {"validation": {}, "additional_discoveries": {}}
                
        except Exception as e:
            logger.error(f"Error processing discovery response: {e}")
            return {"validation": {}, "additional_discoveries": {}}
    
    def _combine_github_repos(self, regex_repos: List[GitHubInfo], llm_result: Dict[str, Any]) -> List[GitHubInfo]:
        """
        Combine regex GitHub repos with LLM discoveries and apply validation.
        
        Args:
            regex_repos: GitHub repos found by regex
            llm_result: LLM discovery and validation results
            
        Returns:
            Combined and validated GitHub repos
        """
        combined_repos = []
        
        # Process regex repos with validation
        validation = llm_result.get("validation", {})
        github_validation = validation.get("github_repos", [])
        
        for repo in regex_repos:
            # Check if LLM validated this repo
            is_valid = True
            for val in github_validation:
                if val.get("url") == repo.url and not val.get("valid", True):
                    is_valid = False
                    logger.info(f"LLM flagged GitHub repo as invalid: {repo.url} - {val.get('reason', 'No reason')}")
                    break
            
            if is_valid:
                combined_repos.append(repo)
        
        # Add LLM discoveries
        discoveries = llm_result.get("additional_discoveries", {})
        new_github_repos = discoveries.get("github_repos", [])
        
        for discovery in new_github_repos:
            try:
                # Parse discovered repo
                url = discovery.get("url", "")
                mention = discovery.get("mention", "")
                confidence = discovery.get("confidence", 0.5)
                
                # Only add high-confidence discoveries
                if confidence >= 0.7 and url:
                    # Extract owner/repo from URL
                    import re
                    match = re.search(r'github\.com/([^/]+)/([^/]+)', url)
                    if match:
                        owner, repo_name = match.groups()
                        repo_name = repo_name.rstrip('.,;!?')  # Clean trailing punctuation
                        
                        # Check if not already exists
                        exists = any(r.url == url for r in combined_repos)
                        if not exists:
                            github_info = GitHubInfo(
                                url=url,
                                owner=owner,
                                repository=repo_name,
                                description=f"Discovered by LLM: {mention}"
                            )
                            combined_repos.append(github_info)
                            logger.info(f"Added LLM-discovered GitHub repo: {url}")
                            
            except Exception as e:
                logger.warning(f"Error processing LLM GitHub discovery: {e}")
        
        return combined_repos
    
    def _combine_code_snippets(self, regex_snippets: List[CodeSnippet], llm_result: Dict[str, Any]) -> List[CodeSnippet]:
        """
        Combine regex code snippets with LLM discoveries and apply validation.
        
        Args:
            regex_snippets: Code snippets found by regex
            llm_result: LLM discovery and validation results
            
        Returns:
            Combined and validated code snippets
        """
        combined_snippets = []
        
        # Process regex snippets with validation
        validation = llm_result.get("validation", {})
        code_validation = validation.get("code_snippets", [])
        
        for i, snippet in enumerate(regex_snippets):
            # Check if LLM validated this snippet
            is_valid = True
            for val in code_validation:
                if val.get("snippet_id") == i and not val.get("valid", True):
                    is_valid = False
                    logger.info(f"LLM flagged code snippet as invalid: {snippet.content[:50]}... - {val.get('reason', 'No reason')}")
                    break
            
            if is_valid:
                combined_snippets.append(snippet)
        
        # Add LLM discoveries
        discoveries = llm_result.get("additional_discoveries", {})
        new_code_snippets = discoveries.get("code_snippets", [])
        
        for discovery in new_code_snippets:
            try:
                content = discovery.get("content", "")
                context = discovery.get("context", "")
                location = discovery.get("location", "")
                confidence = discovery.get("confidence", 0.5)
                
                # Only add high-confidence discoveries
                if confidence >= 0.7 and content and len(content.strip()) >= self.config.min_code_snippet_length:
                    # Detect language for new snippet
                    detected_language = self._detect_programming_language(content)
                    
                    # Check if not already exists (avoid duplicates)
                    content_normalized = content.strip().lower()
                    exists = any(s.content.strip().lower() == content_normalized for s in combined_snippets)
                    
                    if not exists:
                        snippet = CodeSnippet(
                            content=content.strip(),
                            language=detected_language,
                            context=context,
                            start_position=-1,  # LLM discoveries don't have positions
                            end_position=-1,
                            relevance_score=0.0,  # Will be set by analysis
                            description=f"Discovered by LLM in {location}"
                        )
                        combined_snippets.append(snippet)
                        logger.info(f"Added LLM-discovered code snippet from {location}")
                        
            except Exception as e:
                logger.warning(f"Error processing LLM code discovery: {e}")
        
        return combined_snippets
    
    def _combine_external_links(self, regex_links: List[ExternalLink], llm_result: Dict[str, Any]) -> List[ExternalLink]:
        """
        Combine regex external links with LLM discoveries and apply validation.
        
        Args:
            regex_links: External links found by regex
            llm_result: LLM discovery and validation results
            
        Returns:
            Combined and validated external links
        """
        combined_links = []
        
        # Process regex links with validation
        validation = llm_result.get("validation", {})
        link_validation = validation.get("external_links", [])
        
        for link in regex_links:
            # Check if LLM validated this link
            is_valid = True
            for val in link_validation:
                if val.get("url") == link.url and not val.get("valid", True):
                    is_valid = False
                    logger.info(f"LLM flagged external link as invalid: {link.url} - {val.get('reason', 'No reason')}")
                    break
            
            if is_valid:
                combined_links.append(link)
        
        # Add LLM discoveries
        discoveries = llm_result.get("additional_discoveries", {})
        new_external_links = discoveries.get("external_links", [])
        
        for discovery in new_external_links:
            try:
                url = discovery.get("url", "")
                mention = discovery.get("mention", "")
                context = discovery.get("context", "")
                link_type = discovery.get("type", "other")
                confidence = discovery.get("confidence", 0.5)
                
                # Only add high-confidence discoveries
                if confidence >= 0.7 and url:
                    # Check if not already exists
                    exists = any(l.url == url for l in combined_links)
                    if not exists:
                        # Classify link type
                        try:
                            classified_type = LinkType(link_type.lower())
                        except ValueError:
                            classified_type = self._classify_link_type(url)
                        
                        link = ExternalLink(
                            url=url,
                            link_type=classified_type,
                            context=context,
                            description=f"Discovered by LLM: {mention}",
                            relevance_score=0.0  # Will be set by analysis
                        )
                        combined_links.append(link)
                        logger.info(f"Added LLM-discovered external link: {url}")
                        
            except Exception as e:
                logger.warning(f"Error processing LLM link discovery: {e}")
        
        return combined_links
    
    async def _validate_external_links_enhanced(self, links: List[ExternalLink]) -> List[ExternalLink]:
        """
        Validate external links using comprehensive link validation system.
        
        Args:
            links: List of external links to validate
            
        Returns:
            List of validated links with enhanced metadata
        """
        if not links:
            return links
        
        try:
            # Extract URLs for batch validation
            urls = [link.url for link in links]
            
            # Use the comprehensive link validator
            validation_results = await self.link_validator.validate_links_batch(
                urls, max_concurrent=5  # Allow more concurrent for general web links
            )
            
            # Update links with validation results and enhanced metadata
            for link, result in zip(links, validation_results):
                link.is_accessible = result.is_accessible
                
                # Enhance link with web metadata if available
                if result.web_metadata:
                    web_meta = result.web_metadata
                    
                    # Update title if available and not already set
                    if web_meta.title and not link.title:
                        link.title = web_meta.title
                    
                    # Update description if available and not already set
                    if web_meta.description and not link.description:
                        link.description = web_meta.description
                    
                    # Add domain information if useful
                    if web_meta.domain and web_meta.domain not in (link.description or ""):
                        domain_info = f" (from {web_meta.domain})"
                        if link.description:
                            link.description += domain_info
                        else:
                            link.description = f"External resource{domain_info}"
                    
                    # Use Open Graph title/description if better than meta
                    if web_meta.og_title and len(web_meta.og_title) > len(link.title or ""):
                        link.title = web_meta.og_title
                    
                    if web_meta.og_description and len(web_meta.og_description) > len(link.description or ""):
                        link.description = web_meta.og_description
                
                if result.is_accessible:
                    logger.debug(f"External link validated with metadata: {link.url}")
                else:
                    logger.warning(f"External link not accessible: {link.url} - {result.error_message}")
            
            logger.info(f"Enhanced validation completed for {len(links)} external links")
            return links
            
        except Exception as e:
            logger.error(f"Error during enhanced link validation: {e}")
            # Return links with is_accessible=False if validation fails
            for link in links:
                link.is_accessible = False
            return links
    
    async def _analyze_code_snippets(self, snippets: List[CodeSnippet], text_content: str) -> List[CodeSnippet]:
        """
        Analyze code snippets using the dedicated CodeAnalysisLLM module.
        
        Args:
            snippets: List of code snippets to analyze
            text_content: Full text content for context
            
        Returns:
            List of analyzed code snippets with enhanced metadata
        """
        if not snippets:
            return snippets
        
        try:
            # Prepare snippets for the code analysis module
            snippet_data = []
            for i, snippet in enumerate(snippets):
                snippet_data.append({
                    "id": i,
                    "content": snippet.content,
                    "context": snippet.context
                })
            
            # Use the dedicated code analysis module
            analysis_results = await self.code_analyzer.analyze_code_batch(
                code_snippets=snippet_data,
                publication_context=text_content
            )
            
            # Update snippets with analysis results
            analyzed_snippets = []
            for i, snippet in enumerate(snippets):
                if i < len(analysis_results):
                    result = analysis_results[i]
                    
                    # Update snippet with enhanced analysis data
                    snippet.relevance_score = result.relevance_score
                    snippet.description = result.description
                    snippet.purpose = result.purpose.value
                    
                    # Update language if the analysis provided better detection
                    if result.language_confidence > 0.7:
                        try:
                            snippet.language = CodeType(result.language.value)
                        except ValueError:
                            pass  # Keep original language if conversion fails
                    
                    analyzed_snippets.append(snippet)
                else:
                    # Fallback for missing analysis
                    snippet.relevance_score = 5.0
                    analyzed_snippets.append(snippet)
            
            logger.info(f"Successfully analyzed {len(analyzed_snippets)} code snippets using CodeAnalysisLLM")
            return analyzed_snippets
            
        except Exception as e:
            logger.error(f"Error analyzing code snippets with CodeAnalysisLLM: {e}")
            # Return original snippets with default relevance as fallback
            for snippet in snippets:
                snippet.relevance_score = 5.0  # Default moderate relevance
            return snippets
    
    async def _analyze_external_links(self, links: List[ExternalLink], text_content: str) -> List[ExternalLink]:
        """
        Analyze external links using LLM for relevance and categorization.
        
        Args:
            links: List of external links to analyze
            text_content: Full text content for context
            
        Returns:
            List of analyzed external links
        """
        if not links:
            return links
        
        try:
            # Prepare links for LLM analysis
            link_data = []
            for link in links:
                link_data.append({
                    "url": link.url,
                    "initial_type": link.link_type.value,
                    "context": link.context
                })
            
            # Get analysis prompt
            prompt_template = self.llm_service.get_prompt_template("link_analysis")
            prompt = prompt_template.render(
                links=str(link_data),
                paper_context=text_content[:1000]  # First 1000 chars for context
            )
            
            # Call LLM
            response = await self.llm_service.generate(
                prompt=prompt,
                parameters={
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
            )
            
            # Process LLM response
            analyzed_links = self._process_link_analysis_response(response, links)
            return analyzed_links
            
        except Exception as e:
            logger.error(f"Error analyzing external links: {e}")
            # Return original links with default relevance
            for link in links:
                link.relevance_score = 5.0  # Default moderate relevance
            return links
    

    
    def _process_link_analysis_response(self, llm_response: Dict[str, Any], links: List[ExternalLink]) -> List[ExternalLink]:
        """
        Process LLM response for link analysis.
        
        Args:
            llm_response: Response from LLM
            links: Original external links
            
        Returns:
            Updated external links with analysis results
        """
        try:
            # Extract text from LLM response (chat/completions first, fallback to legacy completions)
            response_text = ""
            if "choices" in llm_response and llm_response["choices"]:
                first_choice = llm_response["choices"][0]
                message = first_choice.get("message") or {}
                response_text = (message.get("content") or "").strip()
                if not response_text:
                    response_text = first_choice.get("text", "").strip()
            
            # Parse JSON response
            import json
            try:
                json_match = re.search(r'\[\s*{.*?}\s*\]', response_text, re.DOTALL)
                if not json_match:
                    logger.warning("No JSON found in LLM response")
                    return links
                json_data = json_match.group()

                analyses = json.loads(json_data)
                if not isinstance(analyses, list):
                    analyses = [analyses]
            except (json.JSONDecodeError, ValueError):
                logger.warning("Failed to parse link analysis response")
                return links
            
            # Apply analysis results
            for analysis in analyses:
                if isinstance(analysis, dict) and "url" in analysis:
                    url = analysis["url"]
                    # Find matching link (remove trailing punctuation for comparison)
                    clean_url = url.rstrip('.,;!?')
                    for link in links:
                        clean_link_url = link.url.rstrip('.,;!?')
                        if clean_link_url == clean_url:
                            link.relevance_score = float(analysis.get("relevance_score", 5.0))
                            link.description = analysis.get("description", "")
                            link.title = analysis.get("title", "")
                            
                            # Update link type if LLM provided better classification
                            llm_category = analysis.get("category", "").lower()
                            try:
                                link.link_type = LinkType(llm_category)
                            except ValueError:
                                pass  # Keep original classification
                            break
            
            return links
            
        except Exception as e:
            logger.error(f"Error processing link analysis response: {e}")
            return links
    
    async def _validate_github_repositories(self, repos: List[GitHubInfo]) -> List[GitHubInfo]:
        """
        Validate GitHub repositories by checking their accessibility.
        
        Args:
            repos: List of GitHub repositories to validate
            
        Returns:
            List of repositories with is_valid field updated
        """
        if not repos:
            return repos
        
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                for repo in repos:
                    try:
                        # Check if repository exists by making a HEAD request
                        response = await client.head(repo.url)
                        repo.is_valid = response.status_code == 200
                    except Exception as e:
                        logger.warning(f"Failed to validate GitHub repository {repo.url}: {e}")
                        repo.is_valid = False
            
            return repos
            
        except Exception as e:
            logger.error(f"Error validating GitHub repositories: {e}")
            # Set all as invalid if validation fails
            for repo in repos:
                repo.is_valid = False
            return repos
    
    async def _validate_github_repositories_enhanced(self, repos: List[GitHubInfo]) -> List[GitHubInfo]:
        """
        Validate GitHub repositories using comprehensive link validation system.
        
        Args:
            repos: List of GitHub repositories to validate
            
        Returns:
            List of validated repositories with enhanced metadata
        """
        if not repos:
            return repos
        
        try:
            # Extract URLs for batch validation
            urls = [repo.url for repo in repos]
            
            # Use the comprehensive link validator
            validation_results = await self.link_validator.validate_links_batch(
                urls, max_concurrent=3  # Conservative for GitHub API
            )
            
            # Update repositories with validation results and enhanced metadata
            for repo, result in zip(repos, validation_results):
                repo.is_valid = result.is_accessible
                
                # Enhance repository with GitHub metadata if available
                if result.github_metadata:
                    github_meta = result.github_metadata
                    repo.description = github_meta.description or repo.description
                    repo.language = github_meta.language or repo.language
                    repo.stars = github_meta.stars
                    
                    # Store additional metadata in description if not already set
                    if not repo.description and github_meta.description:
                        repo.description = github_meta.description
                    
                    # Add stars info to description if significant
                    if github_meta.stars and github_meta.stars > 10:
                        star_info = f" ({github_meta.stars} stars)"
                        if repo.description and star_info not in repo.description:
                            repo.description += star_info
                        elif not repo.description:
                            repo.description = f"GitHub repository{star_info}"
                
                if result.is_accessible:
                    logger.debug(f"GitHub repository validated with metadata: {repo.url}")
                else:
                    logger.warning(f"GitHub repository not accessible: {repo.url} - {result.error_message}")
            
            logger.info(f"Enhanced validation completed for {len(repos)} GitHub repositories")
            return repos
            
        except Exception as e:
            logger.error(f"Error during enhanced repository validation: {e}")
            # Return repos with is_valid=False if validation fails
            for repo in repos:
                repo.is_valid = False
            return repos


# Workflow integration function
async def code_extraction_agent_step(state: AnalysisState) -> AnalysisState:
    """
    LangGraph step function for code and link extraction.
    
    Args:
        state: Current analysis state
        
    Returns:
        Updated analysis state with extraction results
    """
    logger.info(f"Starting code extraction step for publication: {state.publication_id}")
    
    try:
        # Initialize the extraction agent
        # Note: In a real implementation, these would be injected dependencies
        from ..services.llm_service import LLMService
        llm_service = LLMService()  # This would be properly initialized
        agent = CodeExtractionAgent(llm_service)
        
        # Perform extraction
        result = await agent.extract_code_and_links(state)
        
        # Convert ExtractionResult to state-compatible data structures
        from ..workflows.state_converters import convert_extraction_result_to_state
        
        converted_data = convert_extraction_result_to_state(result)
        
        # Update state with extraction results
        state.update_extraction_results(
            code_snippets=converted_data["code_snippets"],
            external_links=converted_data["external_links"],
            github_repos=converted_data["github_repos"],
            metadata=converted_data["metadata"]
        )
        
        # Validate data integrity
        validation_errors = state.validate_extraction_data_integrity()
        if validation_errors:
            logger.warning(f"Data integrity validation found issues: {validation_errors}")
            # Add validation errors to extraction metadata
            if state.extraction_metadata:
                state.extraction_metadata.extraction_errors.extend(validation_errors)
        
        # Update workflow step
        state.update_step("code_extraction_completed")
        
        logger.info(
            f"Code extraction completed successfully: {len(converted_data['code_snippets'])} code snippets, "
            f"{len(converted_data['external_links'])} external links, {len(converted_data['github_repos'])} GitHub repos"
        )
        
        # Clean up resources
        await agent.close()
        
        return state
        
    except Exception as e:
        logger.error(f"Error in code extraction step: {e}")
        state.update_step("code_extraction_failed", str(e))
        return state 