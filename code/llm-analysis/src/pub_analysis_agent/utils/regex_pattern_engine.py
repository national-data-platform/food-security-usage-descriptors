"""
Regex Pattern Engine for Code and Link Extraction.

This module provides a comprehensive regex pattern engine for extracting
GitHub URLs, code blocks, and external links from publication text with
high accuracy and metadata extraction capabilities.
"""

import logging
import re
import urllib.parse
from typing import Dict, List, Optional, Tuple, Set, Any, Match
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class PatternType(Enum):
    """Types of patterns supported by the engine."""
    GITHUB_URL = "github_url"
    CODE_BLOCK = "code_block"
    EXTERNAL_LINK = "external_link"
    PROGRAMMING_LANGUAGE = "programming_language"
    INLINE_CODE = "inline_code"


@dataclass
class PatternMatch:
    """Result of a pattern match."""
    pattern_type: PatternType
    full_match: str
    start_position: int
    end_position: int
    groups: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GitHubURLInfo:
    """Detailed information extracted from GitHub URLs."""
    url: str
    owner: str
    repository: str
    path: Optional[str] = None
    branch: Optional[str] = None
    commit_hash: Optional[str] = None
    line_numbers: Optional[Tuple[int, int]] = None
    file_extension: Optional[str] = None
    is_raw_content: bool = False
    is_gist: bool = False
    is_wiki: bool = False
    is_issue: bool = False
    is_pull_request: bool = False
    issue_number: Optional[int] = None
    pull_request_number: Optional[int] = None


@dataclass
class CodeBlockInfo:
    """Information about extracted code blocks."""
    content: str
    block_type: str  # "fenced", "indented", "latex", "inline"
    start_position: int
    end_position: int
    line_count: int
    character_count: int
    language: Optional[str] = None
    has_language_hint: bool = False
    is_complete: bool = True


@dataclass
class ExternalLinkInfo:
    """Information about external links."""
    url: str
    domain: str
    path: str
    query_params: Dict[str, str] = field(default_factory=dict)
    fragment: Optional[str] = None
    protocol: str = "https"
    port: Optional[int] = None
    is_relative: bool = False


class RegexPatternEngine:
    """
    Comprehensive regex pattern engine for extracting code and links from text.
    
    This engine provides robust pattern matching for:
    - GitHub URLs (repositories, gists, raw content, issues, PRs)
    - Code blocks (markdown, indented, LaTeX, inline)
    - External links (various protocols and formats)
    - Programming language detection
    """
    
    def __init__(self) -> None:
        """Initialize the regex pattern engine."""
        self._compile_all_patterns()
        logger.info("RegexPatternEngine initialized with comprehensive patterns")
    
    def _compile_all_patterns(self) -> None:
        """Compile all regex patterns for the engine."""
        self._compile_github_patterns()
        self._compile_code_block_patterns()
        self._compile_external_link_patterns()
        self._compile_language_patterns()
        self._compile_validation_patterns()
    
    def _compile_github_patterns(self) -> None:
        """Compile GitHub URL patterns with comprehensive coverage."""
        self.github_patterns = {
            # GitHub issues and pull requests (highest priority to avoid conflicts)
            "issue_pr": re.compile(
                r'https?://(?:www\.)?github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+)'
                r'/(?:issues|pull)/(\d+)',
                re.IGNORECASE
            ),
            
            # GitHub wiki URLs
            "wiki": re.compile(
                r'https?://(?:www\.)?github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+)'
                r'/wiki/([^#\s]+)',
                re.IGNORECASE
            ),
            
            # GitHub releases
            "release": re.compile(
                r'https?://(?:www\.)?github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+)'
                r'/releases/(?:tag|download)/([^#\s]+)',
                re.IGNORECASE
            ),
            
            # GitHub commit URLs
            "commit": re.compile(
                r'https?://(?:www\.)?github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+)'
                r'/commit/([a-f0-9]{7,40})',
                re.IGNORECASE
            ),
            
            # GitHub raw content URLs
            "raw_content": re.compile(
                r'https?://raw\.githubusercontent\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+)'
                r'/([a-zA-Z0-9._/-]+)/([^#\s]+)'
                r'(?:#L(\d+)(?:-L(\d+))?)?',
                re.IGNORECASE
            ),
            
            # GitHub gist URLs
            "gist": re.compile(
                r'https?://gist\.github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9]+)'
                r'(?:/([a-zA-Z0-9._-]+))?',
                re.IGNORECASE
            ),
            
            # GitHub search URLs
            "search": re.compile(
                r'https?://(?:www\.)?github\.com/search\?q=([^&\s]+)',
                re.IGNORECASE
            ),
            
            # Standard GitHub repository URLs (lowest priority)
            "repository": re.compile(
                r'https?://(?:www\.)?github\.com/([a-zA-Z0-9._-]+)/([a-zA-Z0-9._-]+)'
                r'(?:/(?:tree|blob)/([a-zA-Z0-9._-]+)/([^#\s]+))?'
                r'(?:#L(\d+)(?:-L(\d+))?)?'
                r'(?=[\s.,;!?]|$|/)',
                re.IGNORECASE
            )
        }
    
    def _compile_code_block_patterns(self) -> None:
        """Compile code block patterns for various formats."""
        self.code_block_patterns = {
            # Markdown fenced code blocks with language
            "fenced_with_lang": re.compile(
                r'```(\w+)\s*\n(.*?)\n\s*```',
                re.DOTALL | re.MULTILINE
            ),
            
            # Markdown fenced code blocks without language
            "fenced_no_lang": re.compile(
                r'```\s*\n(.*?)\n\s*```',
                re.DOTALL | re.MULTILINE
            ),
            

            
            # Inline code with backticks
            "inline_backticks": re.compile(
                r'`([^`\n]+)`'
            ),
            
            # Inline code with double backticks
            "inline_double_backticks": re.compile(
                r'``([^`\n]+)``'
            ),
            
            # LaTeX code environments
            "latex_listing": re.compile(
                r'\\begin\{(?:lstlisting|verbatim|code)\}(.*?)\\end\{(?:lstlisting|verbatim|code)\}',
                re.DOTALL
            ),
            
            # LaTeX inline code
            "latex_inline": re.compile(
                r'\\texttt\{([^}]+)\}'
            ),
            
            # LaTeX math mode (potential code)
            "latex_math": re.compile(
                r'\$([^$\n]+)\$'
            ),
            
            # LaTeX display math (potential code)
            "latex_display_math": re.compile(
                r'\$\$([^$]+)\$\$',
                re.DOTALL
            ),
            
            # HTML code tags
            "html_code": re.compile(
                r'<code[^>]*>(.*?)</code>',
                re.DOTALL | re.IGNORECASE
            ),
            
            # HTML pre tags
            "html_pre": re.compile(
                r'<pre[^>]*>(.*?)</pre>',
                re.DOTALL | re.IGNORECASE
            )
        }
    
    def _compile_external_link_patterns(self) -> None:
        """Compile external link patterns."""
        self.external_link_patterns = {
            # General HTTP/HTTPS URLs
            "general_url": re.compile(
                r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?',
                re.IGNORECASE
            ),
            
            # DOI links
            "doi": re.compile(
                r'(?:https?://)?(?:dx\.)?doi\.org/([^\s]+)',
                re.IGNORECASE
            ),
            
            # arXiv links
            "arxiv": re.compile(
                r'(?:https?://)?arxiv\.org/(?:abs|pdf)/([^\s]+)',
                re.IGNORECASE
            ),
            
            # PubMed links
            "pubmed": re.compile(
                r'(?:https?://)?(?:www\.)?ncbi\.nlm\.nih\.gov/pubmed/(\d+)',
                re.IGNORECASE
            ),
            
            # Zenodo links
            "zenodo": re.compile(
                r'(?:https?://)?zenodo\.org/record/(\d+)',
                re.IGNORECASE
            ),
            
            # Figshare links
            "figshare": re.compile(
                r'(?:https?://)?figshare\.com/articles/(\d+)',
                re.IGNORECASE
            ),
            
            # Dryad links
            "dryad": re.compile(
                r'(?:https?://)?datadryad\.org/stash/dataset/([^\s]+)',
                re.IGNORECASE
            ),
            
            # Kaggle links
            "kaggle": re.compile(
                r'(?:https?://)?www\.kaggle\.com/([^\s]+)',
                re.IGNORECASE
            ),
            
            # Google Colab links
            "colab": re.compile(
                r'(?:https?://)?colab\.research\.google\.com/([^\s]+)',
                re.IGNORECASE
            ),
            
            # Jupyter notebook links
            "jupyter": re.compile(
                r'(?:https?://)?(?:[^/]+)/notebooks/([^\s]+)',
                re.IGNORECASE
            ),
            
            # Documentation links (common patterns)
            "documentation": re.compile(
                r'(?:https?://)?(?:docs?\.|documentation\.)([^\s]+)',
                re.IGNORECASE
            ),
            
            # API documentation links
            "api_docs": re.compile(
                r'(?:https?://)?(?:api\.|developer\.)([^\s]+)',
                re.IGNORECASE
            )
        }
    
    def _compile_language_patterns(self) -> None:
        """Compile programming language detection patterns."""
        self.language_patterns = {
            "python": [
                re.compile(r'\bimport\s+\w+', re.IGNORECASE),
                re.compile(r'\bfrom\s+\w+\s+import', re.IGNORECASE),
                re.compile(r'\bdef\s+\w+\s*\(', re.IGNORECASE),
                re.compile(r'\bclass\s+\w+\s*\(', re.IGNORECASE),
                re.compile(r'print\s*\(', re.IGNORECASE),
                re.compile(r'\bif\s+__name__\s*==\s*[\'"]__main__[\'"]', re.IGNORECASE),
                re.compile(r'\bwith\s+\w+\s+as\s+\w+', re.IGNORECASE),
                re.compile(r'\btry:\s*\n', re.IGNORECASE),
                re.compile(r'\bexcept\s+\w+:', re.IGNORECASE),
                re.compile(r'\bfinally:', re.IGNORECASE),
            ],
            "r": [
                re.compile(r'\blibrary\s*\(', re.IGNORECASE),
                re.compile(r'\brequire\s*\(', re.IGNORECASE),
                re.compile(r'<-', re.IGNORECASE),
                re.compile(r'\bdata\.frame\s*\(', re.IGNORECASE),
                re.compile(r'\bggplot\s*\(', re.IGNORECASE),
                re.compile(r'\bfunction\s*\(', re.IGNORECASE),
                re.compile(r'\bfor\s*\(', re.IGNORECASE),
                re.compile(r'\bif\s*\(', re.IGNORECASE),
                re.compile(r'\belse\s*{', re.IGNORECASE),
                re.compile(r'\breturn\s*\(', re.IGNORECASE),
            ],
            "sql": [
                re.compile(r'\bSELECT\s+', re.IGNORECASE),
                re.compile(r'\bFROM\s+', re.IGNORECASE),
                re.compile(r'\bWHERE\s+', re.IGNORECASE),
                re.compile(r'\bJOIN\s+', re.IGNORECASE),
                re.compile(r'\bGROUP\s+BY\s+', re.IGNORECASE),
                re.compile(r'\bORDER\s+BY\s+', re.IGNORECASE),
                re.compile(r'\bHAVING\s+', re.IGNORECASE),
                re.compile(r'\bINSERT\s+INTO\s+', re.IGNORECASE),
                re.compile(r'\bUPDATE\s+', re.IGNORECASE),
                re.compile(r'\bDELETE\s+FROM\s+', re.IGNORECASE),
            ],
            "bash": [
                re.compile(r'#!/bin/(?:bash|sh)', re.IGNORECASE),
                re.compile(r'\b(?:ls|cd|mkdir|rm|cp|mv|grep|awk|sed|find|chmod|chown)\s+', re.IGNORECASE),
                re.compile(r'\$\{?\w+\}?', re.IGNORECASE),
                re.compile(r'\bif\s+\[', re.IGNORECASE),
                re.compile(r'\bthen\s*\n', re.IGNORECASE),
                re.compile(r'\bfi\s*$', re.IGNORECASE),
                re.compile(r'\bfor\s+\w+\s+in', re.IGNORECASE),
                re.compile(r'\bdo\s*\n', re.IGNORECASE),
                re.compile(r'\bdone\s*$', re.IGNORECASE),
                re.compile(r'\bwhile\s+read', re.IGNORECASE),
            ],
            "javascript": [
                re.compile(r'\bfunction\s+\w+\s*\(', re.IGNORECASE),
                re.compile(r'\bconst\s+\w+\s*=', re.IGNORECASE),
                re.compile(r'\blet\s+\w+\s*=', re.IGNORECASE),
                re.compile(r'\bvar\s+\w+\s*=', re.IGNORECASE),
                re.compile(r'console\.log\s*\(', re.IGNORECASE),
                re.compile(r'\bif\s*\(', re.IGNORECASE),
                re.compile(r'\belse\s*{', re.IGNORECASE),
                re.compile(r'\bfor\s*\(', re.IGNORECASE),
                re.compile(r'\bwhile\s*\(', re.IGNORECASE),
                re.compile(r'\breturn\s+', re.IGNORECASE),
            ],
            "java": [
                re.compile(r'\bpublic\s+class\s+\w+', re.IGNORECASE),
                re.compile(r'\bpublic\s+static\s+void\s+main', re.IGNORECASE),
                re.compile(r'\bimport\s+java\.', re.IGNORECASE),
                re.compile(r'\bSystem\.out\.println\s*\(', re.IGNORECASE),
                re.compile(r'\bprivate\s+\w+\s+\w+', re.IGNORECASE),
                re.compile(r'\bprotected\s+\w+\s+\w+', re.IGNORECASE),
                re.compile(r'\bpublic\s+\w+\s+\w+\s*\(', re.IGNORECASE),
                re.compile(r'\bthrows\s+\w+', re.IGNORECASE),
                re.compile(r'\btry\s*{', re.IGNORECASE),
                re.compile(r'\bcatch\s*\(', re.IGNORECASE),
            ],
            "cpp": [
                re.compile(r'#include\s*<[^>]+>', re.IGNORECASE),
                re.compile(r'#include\s*"[^"]+"', re.IGNORECASE),
                re.compile(r'\bint\s+main\s*\(', re.IGNORECASE),
                re.compile(r'\bstd::cout\s*<<', re.IGNORECASE),
                re.compile(r'\bstd::cin\s*>>', re.IGNORECASE),
                re.compile(r'\busing\s+namespace\s+std', re.IGNORECASE),
                re.compile(r'\bclass\s+\w+\s*{', re.IGNORECASE),
                re.compile(r'\bpublic:', re.IGNORECASE),
                re.compile(r'\bprivate:', re.IGNORECASE),
                re.compile(r'\bprotected:', re.IGNORECASE),
            ],
            "matlab": [
                re.compile(r'\bfunction\s+\w+\s*\(', re.IGNORECASE),
                re.compile(r'\bend\s*$', re.IGNORECASE),
                re.compile(r'\bfor\s+\w+\s*=', re.IGNORECASE),
                re.compile(r'\bif\s+\w+\s*==', re.IGNORECASE),
                re.compile(r'\belseif\s+\w+\s*==', re.IGNORECASE),
                re.compile(r'\belse\s*$', re.IGNORECASE),
                re.compile(r'\bplot\s*\(', re.IGNORECASE),
                re.compile(r'\bfigure\s*\(', re.IGNORECASE),
                re.compile(r'\bsubplot\s*\(', re.IGNORECASE),
                re.compile(r'\bhold\s+on', re.IGNORECASE),
            ],
            "julia": [
                re.compile(r'\bfunction\s+\w+\s*\(', re.IGNORECASE),
                re.compile(r'\bend\s*$', re.IGNORECASE),
                re.compile(r'\busing\s+\w+', re.IGNORECASE),
                re.compile(r'\bimport\s+\w+', re.IGNORECASE),
                re.compile(r'\bprintln\s*\(', re.IGNORECASE),
                re.compile(r'\bfor\s+\w+\s+in', re.IGNORECASE),
                re.compile(r'\bif\s+\w+\s*==', re.IGNORECASE),
                re.compile(r'\belseif\s+\w+\s*==', re.IGNORECASE),
                re.compile(r'\belse\s*$', re.IGNORECASE),
                re.compile(r'\bstruct\s+\w+', re.IGNORECASE),
            ]
        }
    
    def _compile_validation_patterns(self) -> None:
        """Compile validation patterns for edge cases."""
        self.validation_patterns = {
            # Incomplete GitHub URLs
            "incomplete_github": re.compile(
                r'https?://(?:www\.)?github\.com/[^/\s]*$',
                re.IGNORECASE
            ),
            
            # Invalid file extensions
            "invalid_extension": re.compile(
                r'\.(?:exe|dll|so|dylib|bin|app)$',
                re.IGNORECASE
            ),
            
            # Suspicious patterns
            "suspicious_pattern": re.compile(
                r'(?:javascript:|data:|vbscript:)',
                re.IGNORECASE
            )
        }
    
    def extract_github_urls(self, text: str) -> List[GitHubURLInfo]:
        """
        Extract GitHub URLs from text with detailed metadata.
        
        Args:
            text: Text to search for GitHub URLs
            
        Returns:
            List of GitHubURLInfo objects with extracted metadata
        """
        results = []
        seen_positions = set()  # Track positions to avoid overlapping matches
        
        for pattern_name, pattern in self.github_patterns.items():
            for match in pattern.finditer(text):
                # Check if this position overlaps with a previous match
                start_pos = match.start()
                end_pos = match.end()
                
                # Check for any overlap with existing matches
                overlaps = False
                for existing_start, existing_end in seen_positions:
                    if (start_pos < existing_end and end_pos > existing_start):
                        overlaps = True
                        break
                
                if overlaps:
                    continue
                
                try:
                    github_info = self._parse_github_match(match, pattern_name)
                    if github_info:
                        results.append(github_info)
                        seen_positions.add((start_pos, end_pos))
                except Exception as e:
                    logger.warning(f"Error parsing GitHub URL match: {e}")
                    continue
        
        return results
    
    def _parse_github_match(self, match: Match, pattern_name: str) -> Optional[GitHubURLInfo]:
        """Parse a GitHub URL match into structured information."""
        url = match.group(0)
        groups = match.groups()
        
        if pattern_name == "repository":
            owner, repo, branch, path, line_start, line_end = groups + (None,) * (6 - len(groups))
            return GitHubURLInfo(
                url=url,
                owner=owner,
                repository=repo,
                branch=branch,
                path=path,
                line_numbers=(int(line_start), int(line_end)) if line_start and line_end else None,
                file_extension=Path(path).suffix if path else None
            )
        
        elif pattern_name == "raw_content":
            owner, repo, branch, path, line_start, line_end = groups + (None,) * (6 - len(groups))
            return GitHubURLInfo(
                url=url,
                owner=owner,
                repository=repo,
                branch=branch,
                path=path,
                is_raw_content=True,
                line_numbers=(int(line_start), int(line_end)) if line_start and line_end else None,
                file_extension=Path(path).suffix if path else None
            )
        
        elif pattern_name == "gist":
            owner, gist_id, filename = groups + (None,) * (3 - len(groups))
            return GitHubURLInfo(
                url=url,
                owner=owner,
                repository=gist_id,
                path=filename,
                is_gist=True
            )
        
        elif pattern_name == "issue_pr":
            owner, repo, number = groups
            is_issue = "issues" in url
            return GitHubURLInfo(
                url=url,
                owner=owner,
                repository=repo,
                is_issue=is_issue,
                is_pull_request=not is_issue,
                issue_number=int(number) if is_issue else None,
                pull_request_number=int(number) if not is_issue else None
            )
        
        elif pattern_name == "wiki":
            owner, repo, page = groups
            return GitHubURLInfo(
                url=url,
                owner=owner,
                repository=repo,
                path=page,
                is_wiki=True
            )
        
        elif pattern_name == "release":
            owner, repo, tag = groups
            return GitHubURLInfo(
                url=url,
                owner=owner,
                repository=repo,
                path=tag
            )
        
        elif pattern_name == "commit":
            owner, repo, commit_hash = groups
            return GitHubURLInfo(
                url=url,
                owner=owner,
                repository=repo,
                commit_hash=commit_hash
            )
        
        elif pattern_name == "search":
            query = groups[0]
            return GitHubURLInfo(
                url=url,
                owner="search",
                repository=query
            )
        
        return None
    
    def extract_code_blocks(self, text: str) -> List[CodeBlockInfo]:
        """
        Extract code blocks from text with language detection.
        
        Args:
            text: Text to search for code blocks
            
        Returns:
            List of CodeBlockInfo objects with extracted metadata
        """
        results = []
        seen_positions = set()  # Track positions to avoid overlapping matches
        
        for pattern_name, pattern in self.code_block_patterns.items():
            for match in pattern.finditer(text):
                # Check if this position overlaps with a previous match
                start_pos = match.start()
                end_pos = match.end()
                
                # Check for any overlap with existing matches
                overlaps = False
                for existing_start, existing_end in seen_positions:
                    if (start_pos < existing_end and end_pos > existing_start):
                        overlaps = True
                        break
                
                if overlaps:
                    continue
                
                try:
                    code_info = self._parse_code_block_match(match, pattern_name)
                    if code_info:
                        results.append(code_info)
                        seen_positions.add((start_pos, end_pos))
                except Exception as e:
                    logger.warning(f"Error parsing code block match: {e}")
                    continue
        
        return results
    
    def _parse_code_block_match(self, match: Match, pattern_name: str) -> Optional[CodeBlockInfo]:
        """Parse a code block match into structured information."""
        full_match = match.group(0)
        start_pos = match.start()
        end_pos = match.end()
        
        if pattern_name == "fenced_with_lang":
            language, content = match.groups()
            return CodeBlockInfo(
                content=content,
                language=language,
                block_type="fenced",
                start_position=start_pos,
                end_position=end_pos,
                line_count=content.count('\n') + 1,
                character_count=len(content),
                has_language_hint=True
            )
        
        elif pattern_name == "fenced_no_lang":
            content = match.group(1)
            return CodeBlockInfo(
                content=content,
                language=None,
                block_type="fenced",
                start_position=start_pos,
                end_position=end_pos,
                line_count=content.count('\n') + 1,
                character_count=len(content),
                has_language_hint=False
            )
        
        elif pattern_name == "indented":
            content = match.group(1)
            return CodeBlockInfo(
                content=content,
                language=None,
                block_type="indented",
                start_position=start_pos,
                end_position=end_pos,
                line_count=1,
                character_count=len(content),
                has_language_hint=False
            )
        
        elif pattern_name in ["inline_backticks", "inline_double_backticks"]:
            content = match.group(1)
            return CodeBlockInfo(
                content=content,
                language=None,
                block_type="inline",
                start_position=start_pos,
                end_position=end_pos,
                line_count=1,
                character_count=len(content),
                has_language_hint=False
            )
        
        elif pattern_name == "latex_listing":
            content = match.group(1)
            return CodeBlockInfo(
                content=content,
                language="latex",
                block_type="latex",
                start_position=start_pos,
                end_position=end_pos,
                line_count=content.count('\n') + 1,
                character_count=len(content),
                has_language_hint=True
            )
        
        elif pattern_name == "latex_inline":
            content = match.group(1)
            return CodeBlockInfo(
                content=content,
                language="latex",
                block_type="inline",
                start_position=start_pos,
                end_position=end_pos,
                line_count=1,
                character_count=len(content),
                has_language_hint=True
            )
        
        elif pattern_name in ["html_code", "html_pre"]:
            content = match.group(1)
            return CodeBlockInfo(
                content=content,
                language="html",
                block_type="html",
                start_position=start_pos,
                end_position=end_pos,
                line_count=content.count('\n') + 1,
                character_count=len(content),
                has_language_hint=True
            )
        
        return None
    
    def extract_external_links(self, text: str) -> List[ExternalLinkInfo]:
        """
        Extract external links from text with metadata.
        
        Args:
            text: Text to search for external links
            
        Returns:
            List of ExternalLinkInfo objects with extracted metadata
        """
        results = []
        
        for pattern_name, pattern in self.external_link_patterns.items():
            for match in pattern.finditer(text):
                try:
                    link_info = self._parse_external_link_match(match, pattern_name)
                    if link_info:
                        results.append(link_info)
                except Exception as e:
                    logger.warning(f"Error parsing external link match: {e}")
                    continue
        
        # Remove duplicates while preserving order
        seen_urls = set()
        unique_results = []
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results
    
    def _parse_external_link_match(self, match: Match, pattern_name: str) -> Optional[ExternalLinkInfo]:
        """Parse an external link match into structured information."""
        url = match.group(0)
        
        try:
            parsed_url = urllib.parse.urlparse(url)
            
            # Parse query parameters
            query_params = {}
            if parsed_url.query:
                query_params = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
            
            return ExternalLinkInfo(
                url=url,
                domain=parsed_url.netloc,
                path=parsed_url.path,
                query_params=query_params,
                fragment=parsed_url.fragment,
                protocol=parsed_url.scheme,
                port=parsed_url.port,
                is_relative=not parsed_url.scheme
            )
        except Exception as e:
            logger.warning(f"Error parsing URL {url}: {e}")
            return None
    
    def detect_programming_language(self, code: str, language_hint: Optional[str] = None) -> Optional[str]:
        """
        Detect programming language from code content.
        
        Args:
            code: Code content to analyze
            language_hint: Optional language hint from markdown or other sources
            
        Returns:
            Detected programming language or None if uncertain
        """
        if language_hint:
            language_hint = language_hint.lower()
            if language_hint in self.language_patterns:
                return language_hint
        
        # Score each language based on pattern matches
        language_scores = {}
        
        for language, patterns in self.language_patterns.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(code)
                score += len(matches)
            language_scores[language] = score
        
        # Return the language with the highest score, if any
        if language_scores:
            best_language = max(language_scores.items(), key=lambda x: x[1])
            if best_language[1] > 0:  # Only return if we found some matches
                return best_language[0]
        
        return None
    
    def validate_url(self, url: str) -> bool:
        """
        Validate URL format and check for suspicious patterns.
        
        Args:
            url: URL to validate
            
        Returns:
            True if URL appears valid, False otherwise
        """
        # Check for suspicious patterns
        for pattern_name, pattern in self.validation_patterns.items():
            if pattern.search(url):
                logger.warning(f"URL {url} matches suspicious pattern: {pattern_name}")
                return False
        
        # Basic URL format validation
        try:
            parsed = urllib.parse.urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Check for invalid file extensions
            if self.validation_patterns["invalid_extension"].search(url):
                return False
                
            return True
        except Exception:
            return False
    
    def extract_all_patterns(self, text: str) -> Dict[str, List[Any]]:
        """
        Extract all patterns from text in a single pass.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary containing all extracted patterns
        """
        return {
            "github_urls": self.extract_github_urls(text),
            "code_blocks": self.extract_code_blocks(text),
            "external_links": self.extract_external_links(text)
        }
    
    def get_pattern_statistics(self, text: str) -> Dict[str, int]:
        """
        Get statistics about patterns found in text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with pattern counts
        """
        patterns = self.extract_all_patterns(text)
        
        return {
            "total_github_urls": len(patterns["github_urls"]),
            "total_code_blocks": len(patterns["code_blocks"]),
            "total_external_links": len(patterns["external_links"]),
            "fenced_code_blocks": len([cb for cb in patterns["code_blocks"] if cb.block_type == "fenced"]),
            "inline_code": len([cb for cb in patterns["code_blocks"] if cb.block_type == "inline"]),
            "latex_code": len([cb for cb in patterns["code_blocks"] if cb.block_type == "latex"]),
            "html_code": len([cb for cb in patterns["code_blocks"] if cb.block_type == "html"]),
            "repositories": len([gh for gh in patterns["github_urls"] if not gh.is_gist and not gh.is_wiki and not gh.is_issue and not gh.is_pull_request]),
            "gists": len([gh for gh in patterns["github_urls"] if gh.is_gist]),
            "issues_prs": len([gh for gh in patterns["github_urls"] if gh.is_issue or gh.is_pull_request])
        } 