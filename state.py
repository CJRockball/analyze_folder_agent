"""State schema for the project analysis agent."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from typing_extensions import TypedDict


class FileMetadata(BaseModel):
    """Metadata for a single file."""
    path: str
    filename: str
    extension: str
    size: int
    modified_time: str
    creation_time: Optional[str] = None
    git_first_commit: Optional[str] = None


class FileAnalysis(BaseModel):
    """Analysis results for a single file."""
    file_path: str
    file_type: str
    purpose: str
    key_components: List[str]
    imports: List[str]
    complexity_score: Optional[int] = None
    quality_notes: List[str]


class ProjectInsights(BaseModel):
    """High-level project insights."""
    research_topics: List[str]
    frameworks_used: List[str]
    estimated_timeline: str
    project_type: str
    overall_quality: str
    duplication_analysis: Optional[Dict[str, Any]] = None


class AnalysisState(TypedDict):
    """Main state for the analysis workflow."""
    target_directory: str
    discovered_files: List[FileMetadata]
    file_analyses: List[FileAnalysis]
    project_insights: ProjectInsights
    analysis_report: str
    error_messages: List[str]