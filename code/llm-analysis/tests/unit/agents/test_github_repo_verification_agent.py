import io
import zipfile
from typing import List

import pytest

from pub_analysis_agent.agents.github_repo_verification_agent import (
    GitHubRepositoryVerificationAgent,
    GitHubVerificationConfig,
)
from pub_analysis_agent.workflows.state_models import (
    AnalysisState,
    ExtractedGitHubRepository,
)
from pub_analysis_agent.services.link_validator import (
    LinkValidator,
    LinkValidationResult,
    GitHubMetadata,
)
from pub_analysis_agent.services.llm_service import PromptTemplate


class FakeLLMService:
    def __init__(self) -> None:
        self._prompts = {}

    def add_prompt_template(self, template: PromptTemplate) -> None:
        self._prompts[template.name] = template

    def get_prompt_template(self, name: str) -> PromptTemplate:
        return self._prompts[name]

    async def generate(self, prompt: str, parameters: dict) -> dict:
        # Return a minimal chat-like response including JSON payload
        content = '{"alignment_score": 8.5, "key_signals": ["dataset referenced", "pipeline present"], "code_description": "Python analysis pipeline for data processing", "data_sanitization": "Found data cleaning functions for missing values and outliers", "dataset_joins": "Evidence of merging demographic and examination data", "data_sanitization_code": "def clean_data(df):\\n    df = df.dropna()\\n    df = df[df[\'value\'] > 0]\\n    return df", "dataset_joins_code": "merged_df = pd.merge(demographic_data, exam_data, on=\'patient_id\', how=\'inner\')"}'
        return {"choices": [{"message": {"content": content}}]}


def _make_zip_bytes(files: dict[str, str]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, mode="w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return bio.getvalue()


@pytest.mark.asyncio
async def test_verify_returns_empty_when_no_repos() -> None:
    llm = FakeLLMService()
    agent = GitHubRepositoryVerificationAgent(llm)
    state = AnalysisState(publication_id="pub-1", raw_text="Some publication text")

    results = await agent.verify_github_repositories(state)

    assert isinstance(results, list)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_verify_success_with_zip_and_llm(monkeypatch) -> None:
    llm = FakeLLMService()
    agent = GitHubRepositoryVerificationAgent(llm, link_validator=LinkValidator())

    # Monkeypatch link validator batch to return accessible repo with metadata
    async def fake_validate_batch(urls: List[str], max_concurrent: int = 3):
        return [
            LinkValidationResult(
                url=urls[0],
                is_valid=True,
                is_accessible=True,
                validation_time=0.1,
                github_metadata=GitHubMetadata(
                    url=urls[0],
                    name="demo",
                    full_name="owner/demo",
                    language="Python",
                    stars=42,
                    default_branch="main",
                ),
            )
        ]

    monkeypatch.setattr(agent.link_validator, "validate_links_batch", fake_validate_batch)

    # Monkeypatch download to return a small ZIP with relevant content
    zip_bytes = _make_zip_bytes(
        {
            "demo-main/README.md": "This project uses NHANES dataset. Methods, results, evaluation pipeline benchmark.",
            "demo-main/src/main.py": "print('analysis')\n# preprocessing modeling visualization\n",
        }
    )
    async def fake_try_download_zip(client, url: str):
        return zip_bytes

    monkeypatch.setattr(agent, "_try_download_zip", fake_try_download_zip)

    # Build state with one repo and rich publication context
    state = AnalysisState(
        publication_id="pub-2",
        raw_text=(
            "This study analyzes NHANES with methods, results, experiments, evaluation, reproducibility, "
            "pipeline and benchmark using Python tooling."
        ),
    )
    state.add_extracted_github_repo(
        ExtractedGitHubRepository(url="https://github.com/owner/demo", owner="owner", repository="demo")
    )

    results = await agent.verify_github_repositories(state)

    assert len(results) == 1
    r = results[0]
    assert r.downloaded is True
    assert r.files_analyzed_count > 0
    assert 0.0 <= r.alignment_score <= 10.0
    assert r.passed is True
    # Expect signals augmented by LLM
    assert any("LLM:" in s for s in r.key_findings)
    # Check new fields are populated
    assert r.code_description != ""
    assert r.data_sanitization != ""
    assert r.dataset_joins != ""
    assert r.data_sanitization_code != ""
    assert r.dataset_joins_code != ""


@pytest.mark.asyncio
async def test_verify_handles_llm_failure(monkeypatch) -> None:
    class FailingLLM(FakeLLMService):
        async def generate(self, prompt: str, parameters: dict) -> dict:  # type: ignore[override]
            raise RuntimeError("LLM unreachable")

    llm = FailingLLM()
    agent = GitHubRepositoryVerificationAgent(llm, link_validator=LinkValidator(), config=GitHubVerificationConfig(alignment_threshold=3.0))

    async def fake_validate_batch(urls: List[str], max_concurrent: int = 3):
        return [
            LinkValidationResult(
                url=urls[0],
                is_valid=True,
                is_accessible=True,
                validation_time=0.1,
                github_metadata=GitHubMetadata(
                    url=urls[0],
                    name="demo",
                    full_name="owner/demo",
                    language="Python",
                    stars=10,
                    default_branch="main",
                ),
            )
        ]

    monkeypatch.setattr(agent.link_validator, "validate_links_batch", fake_validate_batch)

    zip_bytes = _make_zip_bytes({"demo-main/README.md": "NHANES methods results pipeline"})

    async def fake_try_download_zip(client, url: str):
        return zip_bytes

    monkeypatch.setattr(agent, "_try_download_zip", fake_try_download_zip)

    state = AnalysisState(
        publication_id="pub-3",
        raw_text="NHANES methods results pipeline",
    )
    state.add_extracted_github_repo(
        ExtractedGitHubRepository(url="https://github.com/owner/demo", owner="owner", repository="demo")
    )

    results = await agent.verify_github_repositories(state)

    assert len(results) == 1
    r = results[0]
    assert r.downloaded is True
    # Should still have a heuristic score even if LLM fails
    assert r.alignment_score > 0.0

