"""
GitHub Repository Verification Agent.

This agent verifies GitHub repositories extracted from the publication by:
- Validating accessibility and fetching metadata (default branch, language, stars)
- Downloading the default-branch ZIP (when possible) and inspecting key files
- Using both heuristic and LLM-based analysis to score alignment (0-10) with the publication

Requirements:
- Asynchronous (httpx)
- Resource-safe (size/time limits)
"""

from __future__ import annotations

import io
import logging
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx

from ..services.link_validator import LinkValidator, LinkValidationConfig, LinkValidationResult
from ..services.llm_service import LLMService, PromptTemplate
from ..workflows.state_models import AnalysisState, ExtractedGitHubRepository, RepositoryVerificationResult


logger = logging.getLogger(__name__)


@dataclass
class GitHubVerificationConfig:
    """Configuration for GitHub repository verification.

    Attributes:
        max_zip_size_bytes: Maximum allowed ZIP size (bytes)
        http_timeout: Total HTTP timeout (seconds)
        alignment_threshold: Threshold to consider alignment as passed (0-10)
        inspect_file_patterns: File patterns in the ZIP to inspect
    """

    max_zip_size_bytes: int = 25 * 1024 * 1024
    http_timeout: float = 30.0
    alignment_threshold: float = 6.0
    inspect_file_patterns: Tuple[str, ...] = (
        r"(?i)README(?:\.md|\.rst)?$",
        r"(?i)requirements\.txt$",
        r"(?i)pyproject\.toml$",
        r"(?i)setup\.py$",
        r"(?i)environment\.yml$",
        r"(?i).*\.ipynb$",
        r"(?i).*\.py$",
        # R / statistics
        r"(?i).*\.r$",
        r"(?i).*\.rmd$",
        r"(?i).*\.rmarkdown$",
        r"(?i).*\.sas$",
        r"(?i).*\.sps$",
        r"(?i).*\.do$",
        r"(?i).*\.ado$",
        # Julia / MATLAB / Octave
        r"(?i).*\.jl$",
        r"(?i).*\.m$",
        # SQL / Shell
        r"(?i).*\.sql$",
        r"(?i).*\.sh$",
        r"(?i).*\.bash$",
        # C / C++ / Java / Scala / Fortran
        r"(?i).*\.c$",
        r"(?i).*\.cpp$",
        r"(?i).*\.cc$",
        r"(?i).*\.h$",
        r"(?i).*\.hpp$",
        r"(?i).*\.java$",
        r"(?i).*\.scala$",
        r"(?i).*\.f$",
        r"(?i).*\.f90$",
        r"(?i).*\.f95$",
        # TeX / Markup / Config
        r"(?i).*\.tex$",
        r"(?i).*\.md$",
        r"(?i).*\.yaml$",
        r"(?i).*\.yml$",
        r"(?i)Dockerfile$",
        r"(?i)Makefile$",
    )


class GitHubRepositoryVerificationAgent:
    """Agent that verifies whether repositories match the publication intent.

    Flow:
    1) Validate accessibility and fetch GitHub metadata via LinkValidator
    2) Download the default branch ZIP (when available)
    3) Inspect key files and compute alignment using heuristics and LLM
    """

    def __init__(
        self,
        llm_service: LLMService,
        link_validator: Optional[LinkValidator] = None,
        config: Optional[GitHubVerificationConfig] = None,
    ) -> None:
        self.llm_service = llm_service
        self.link_validator = link_validator or LinkValidator(LinkValidationConfig())
        self.config = config or GitHubVerificationConfig()
        self._setup_prompt_templates()

    def _setup_prompt_templates(self) -> None:
        """Register agent-specific prompts with the LLM service."""
        repo_alignment_prompt = PromptTemplate(
            name="repo_alignment_analysis",
            template="""You are an expert auditor validating whether a GitHub repository matches a research publication.

PUBLICATION CONTEXT (truncated):
{publication_context}

REPOSITORY INFO:
- url: {repo_url}
- owner: {owner}
- name: {repository}
- language: {language}
- stars: {stars}

FILE EXCERPTS (path and first lines; multiple files):
{file_excerpts}

Task: Assess how well the repository's content aligns with the publication's described methods, data, and objectives.
Consider dataset names, methods, pipelines, experiments, evaluation, and reproducibility indicators.

SPECIFIC ANALYSIS REQUIREMENTS:
1. DATA SANITIZATION/FILTERING: Analyze if the code performs any data cleaning, filtering, preprocessing, or sanitization steps on datasets. Look for:
   - Data validation, cleaning, or preprocessing functions
   - Missing value handling, outlier removal, data normalization
   - Data quality checks, filtering conditions, or data transformation steps
   - Evidence of data preparation pipelines

2. DATASET JOINS/INTEGRATION: If multiple datasets are used, analyze evidence of data integration:
   - JOIN operations, merge functions, or data combination logic
   - Key matching, foreign key relationships, or linking strategies
   - Evidence of combining data from different sources or tables
   - Data alignment, synchronization, or cross-referencing between datasets

IMPORTANT: For both data sanitization and dataset joins, if you find evidence, extract the actual code snippets (functions, methods, or code blocks) that implement these features. Include the code in the respective fields below.

Respond strictly as JSON object with fields:
{{
  "alignment_score": <0-10 number>,
  "key_signals": ["<short bullet>", ...],
  "notes": "<concise rationale>",
  "code_description": "<brief description of what the analyzed code does, its main functionality and purpose>",
  "data_sanitization": "<description of any data cleaning, filtering, or preprocessing found in the code, or 'None detected' if not found>",
  "dataset_joins": "<description of any evidence of dataset joins, merges, or data integration, or 'None detected' if not found>",
  "data_sanitization_code": "<actual code snippets showing data sanitization implementation, or empty string if none found>",
  "dataset_joins_code": "<actual code snippets showing dataset joins implementation, or empty string if none found>"
}}""",
            variables=[
                "publication_context",
                "repo_url",
                "owner",
                "repository",
                "language",
                "stars",
                "file_excerpts",
            ],
        )
        self.llm_service.add_prompt_template(repo_alignment_prompt)

    async def verify_github_repositories(self, state: AnalysisState) -> List[RepositoryVerificationResult]:
        """Verify each extracted GitHub repository found in the state.

        Args:
            state: Current analysis state

        Returns:
            List of per-repository verification results
        """
        repos = state.extracted_github_repos or []
        if not repos:
            logger.info("No extracted GitHub repositories. Skipping verification.")
            return []

        # Validar/obter metadados via LinkValidator
        urls = [r.url for r in repos if r.url]
        validation_results = await self.link_validator.validate_links_batch(urls, max_concurrent=3)
        url_to_validation: Dict[str, LinkValidationResult] = {vr.url: vr for vr in validation_results}

        results: List[RepositoryVerificationResult] = []
        async with httpx.AsyncClient(timeout=httpx.Timeout(self.config.http_timeout)) as client:
            for repo in repos:
                try:
                    vr = url_to_validation.get(repo.url)
                    default_branch = None
                    if vr and vr.github_metadata and vr.github_metadata.default_branch:
                        default_branch = vr.github_metadata.default_branch
                    branch = default_branch or repo.branch or "main"

                    # Construir URL de download de ZIP (fallback para master)
                    zip_url = f"https://codeload.github.com/{repo.owner}/{repo.repository}/zip/refs/heads/{branch}"
                    zip_bytes: Optional[bytes] = await self._try_download_zip(client, zip_url)
                    if zip_bytes is None and branch != "master":
                        alt_url = f"https://codeload.github.com/{repo.owner}/{repo.repository}/zip/refs/heads/master"
                        zip_bytes = await self._try_download_zip(client, alt_url)

                    files_text_preview: Dict[str, str] = {}
                    files_analyzed_count = 0
                    downloaded = False

                    if zip_bytes is not None:
                        downloaded = True
                        try:
                            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                                for info in zf.infolist():
                                    # Skip directories and large files (>256KB)
                                    if info.is_dir() or info.file_size > 256 * 1024:
                                        continue
                                    name = info.filename
                                    if self._should_inspect(name):
                                        with zf.open(info) as f:
                                            try:
                                                content = f.read().decode("utf-8", errors="replace")
                                            except Exception:
                                                continue
                                        files_text_preview[name] = content[:8000]
                                        files_analyzed_count += 1
                                        # Limit the number of inspected files
                                        if files_analyzed_count >= 100:
                                            break
                        except Exception as e:
                            logger.warning(f"Failed to inspect ZIP for {repo.url}: {e}")

                    # Heuristic alignment
                    heuristic_alignment_score, matched_terms = self._compute_alignment(
                        publication_text=state.raw_text or "",
                        repo_texts=list(files_text_preview.values()),
                        datasets=[d.name if hasattr(d, 'name') else getattr(d, 'dataset_name', str(d)) for d in state.get_all_datasets()],
                    )

                    # Initialize key_findings list before LLM analysis
                    key_findings = []
                    if matched_terms:
                        key_findings.append(
                            f"Matched terms: {', '.join(sorted(set(matched_terms))[:10])}"
                        )

                    # LLM-based alignment using an agent-specific prompt
                    llm_alignment_score: float = 0.0
                    code_description: str = ""
                    data_sanitization: str = ""
                    dataset_joins: str = ""
                    data_sanitization_code: str = ""
                    dataset_joins_code: str = ""
                    if files_text_preview:
                        try:
                            file_excerpts_lines: List[str] = []
                            for path, content in list(files_text_preview.items())[:15]:
                                short = content.splitlines()
                                head = "\n".join(short[:40])
                                file_excerpts_lines.append(f"- {path}:\n{head}")
                            file_excerpts = "\n\n".join(file_excerpts_lines)

                            lang = None
                            stars = None
                            if repo.language:
                                lang = repo.language
                            if repo.stars is not None:
                                stars = repo.stars

                            prompt = self.llm_service.get_prompt_template("repo_alignment_analysis").render(
                                publication_context=(state.raw_text or ""),
                                repo_url=repo.url,
                                owner=repo.owner,
                                repository=repo.repository,
                                language=lang or "unknown",
                                stars=str(stars) if stars is not None else "unknown",
                                file_excerpts=file_excerpts[:12000],
                            )
                            response = await self.llm_service.generate(
                                prompt=prompt,
                                parameters={"temperature": 0.7, "max_tokens": 4000},
                            )

                            # Extract JSON object from response
                            text = ""
                            if "choices" in response and response["choices"]:
                                msg = response["choices"][0].get("message") or {}
                                text = (msg.get("content") or "").strip() or response["choices"][0].get("text", "")
                            import json, re
                            m = re.search(r"\{[\s\S]*\}", text)
                            if m:
                                data = json.loads(m.group(0))
                                raw_score = float(data.get("alignment_score", 0.0))
                                llm_alignment_score = max(0.0, min(10.0, raw_score))
                                signals = data.get("key_signals", [])
                                if isinstance(signals, list) and signals:
                                    key_findings.extend([f"LLM: {s}" for s in signals[:5]])
                                code_description = data.get("code_description", "")
                                data_sanitization = data.get("data_sanitization", "")
                                dataset_joins = data.get("dataset_joins", "")
                                data_sanitization_code = data.get("data_sanitization_code", "")
                                dataset_joins_code = data.get("dataset_joins_code", "")
                        except Exception as e:
                            logger.warning(f"Agent-level LLM analysis failed for repo {repo.url}: {e}")

                    # Combine heuristic + LLM (balanced)
                    alignment_score = min(10.0, 0.5 * heuristic_alignment_score + 0.5 * llm_alignment_score)

                    passed = alignment_score >= self.config.alignment_threshold
                    
                    if llm_alignment_score > 0:
                        key_findings.append(
                            f"LLM relevance average: {llm_alignment_score:.2f} (0-10)"
                        )
                    if vr and vr.github_metadata and vr.github_metadata.language:
                        key_findings.append(f"Primary language: {vr.github_metadata.language}")

                    result = RepositoryVerificationResult(
                        url=repo.url,
                        owner=repo.owner,
                        repository=repo.repository,
                        default_branch=branch,
                        downloaded=downloaded,
                        files_analyzed_count=files_analyzed_count,
                        alignment_score=round(alignment_score, 2),
                        passed=passed,
                        key_findings=key_findings,
                        code_description=code_description,
                        data_sanitization=data_sanitization,
                        dataset_joins=dataset_joins,
                        data_sanitization_code=data_sanitization_code,
                        dataset_joins_code=dataset_joins_code,
                        error=None,
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error verifying repo {repo.url}: {e}")
                    results.append(
                        RepositoryVerificationResult(
                            url=repo.url,
                            owner=repo.owner,
                            repository=repo.repository,
                            default_branch=repo.branch or "",
                            downloaded=False,
                            files_analyzed_count=0,
                            alignment_score=0.0,
                            passed=False,
                            key_findings=[],
                            code_description="",
                            data_sanitization="",
                            dataset_joins="",
                            data_sanitization_code="",
                            dataset_joins_code="",
                            error=str(e),
                        )
                    )

        return results

    async def _try_download_zip(self, client: httpx.AsyncClient, url: str) -> Optional[bytes]:
        """Attempt to download a ZIP while respecting size limits."""
        try:
            resp = await client.get(url)
            if resp.status_code != 200 or "application/zip" not in resp.headers.get("content-type", "").lower():
                return None
            content_length = resp.headers.get("content-length")
            if content_length is not None and int(content_length) > self.config.max_zip_size_bytes:
                logger.warning(f"ZIP too large at {url}: {content_length} bytes")
                return None
            data = resp.content
            if len(data) > self.config.max_zip_size_bytes:
                logger.warning(f"ZIP exceeded limit after download at {url}")
                return None
            return data
        except Exception:
            return None

    def _should_inspect(self, file_name: str) -> bool:
        """Return True if file matches inspection patterns."""
        return any(re.search(pattern, file_name) for pattern in self.config.inspect_file_patterns)

    def _compute_alignment(
        self,
        publication_text: str,
        repo_texts: List[str],
        datasets: List[str],
    ) -> Tuple[float, List[str]]:
        """Simple heuristic alignment using publication terms and dataset names.

        Strategy:
        - Publication keywords (top-N alphabetic tokens with length >= 5) matched in README/files
        - Validated dataset names matched directly
        - Bonus words: method, result, experiment, evaluation, reproduc, pipeline
        """
        if not repo_texts:
            return 0.0, []

        text_repo = "\n".join(repo_texts).lower()
        matched: List[str] = []

        # datasets
        for ds in datasets:
            if ds and ds.lower() in text_repo:
                matched.append(ds)

        # Publication keywords
        pub_tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{4,}", (publication_text or "").lower())
        # Normalize and take up to 50 distinct terms
        uniq_terms: List[str] = []
        seen = set()
        for t in pub_tokens:
            if t not in seen:
                seen.add(t)
                uniq_terms.append(t)
            if len(uniq_terms) >= 50:
                break
        for term in uniq_terms:
            if term in text_repo:
                matched.append(term)

        # Bonus words
        bonus_terms = [
            "method", "methods", "result", "results", "experiment", "experiments",
            "evaluation", "reproduc", "pipeline", "benchmark",
        ]
        for bt in bonus_terms:
            if bt in text_repo:
                matched.append(bt)

        # Simple scoring: unique matches with saturation at 10
        unique_matches = len(set(matched))
        score = min(10.0, unique_matches * 0.8)
        return score, matched


