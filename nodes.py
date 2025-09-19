"""Processing nodes for the project analysis agent."""

import os
import ast
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List

from state import AnalysisState, FileMetadata, FileAnalysis, ProjectInsights
from perplexity_client import PerplexityClient


def discover_files(state: AnalysisState) -> AnalysisState:
    """Discover and catalog all relevant files in the directory."""
    print(f"🔍 Scanning directory: {state['target_directory']}")
    
    target_extensions = {'.py', '.html', '.css', '.csv', '.parquet', '.h5', '.hdf5', '.txt'}
    discovered_files = []
    
    try:
        for root, _, files in os.walk(state['target_directory']):
            for file in files:
                file_path = Path(root) / file
                extension = file_path.suffix.lower()
                
                if extension in target_extensions:
                    # Get basic metadata
                    stat_info = file_path.stat()
                    
                    # Try to get Git creation date
                    git_date = get_git_creation_date(file_path)
                    
                    metadata = FileMetadata(
                        path=str(file_path),
                        filename=file,
                        extension=extension,
                        size=stat_info.st_size,
                        modified_time=datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                        creation_time=datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                        git_first_commit=git_date
                    )
                    discovered_files.append(metadata)
        
        print(f"📁 Found {len(discovered_files)} relevant files")
        state['discovered_files'] = discovered_files
        
    except Exception as e:
        state['error_messages'].append(f"File discovery error: {str(e)}")
    
    return state


def analyze_python_files(state: AnalysisState) -> AnalysisState:
    """Analyze Python files using AST and Perplexity."""
    print("🐍 Analyzing Python files...")
    
    perplexity = PerplexityClient()
    python_files = [f for f in state['discovered_files'] if f.extension == '.py']
    
    for file_meta in python_files:
        try:
            # Read file content
            with open(file_meta.path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # AST analysis for structure
            tree = ast.parse(content)
            imports = extract_imports(tree)
            components = extract_components(tree)
            complexity = calculate_complexity(tree)
            
            # Perplexity analysis for purpose and quality
            analysis_result = perplexity.analyze_code(content, "python")
            
            file_analysis = FileAnalysis(
                file_path=file_meta.path,
                file_type="Python",
                purpose=f"AST Analysis + AI: {analysis_result[:200]}...",
                key_components=components,
                imports=imports,
                complexity_score=complexity,
                quality_notes=[f"Complexity: {complexity}", "AI analysis included"]
            )
            
            state['file_analyses'].append(file_analysis)
            print(f"  ✅ {file_meta.filename}")
            
        except Exception as e:
            state['error_messages'].append(f"Python analysis error for {file_meta.filename}: {str(e)}")
            print(f"  ❌ {file_meta.filename}: {str(e)}")
    
    return state


def analyze_other_files(state: AnalysisState) -> AnalysisState:
    """Analyze non-Python files for context."""
    print("📄 Analyzing other files...")
    
    other_files = [f for f in state['discovered_files'] if f.extension != '.py']
    
    for file_meta in other_files:
        try:
            # Basic file analysis
            file_type = get_file_type_description(file_meta.extension)
            purpose = f"{file_type} file - {file_meta.size} bytes"
            
            # Quick content peek for text files
            if file_meta.extension in {'.html', '.css', '.txt'}:
                with open(file_meta.path, 'r', encoding='utf-8', errors='ignore') as f:
                    sample = f.read(500)
                    purpose += f" | Sample: {sample[:100]}..."
            
            file_analysis = FileAnalysis(
                file_path=file_meta.path,
                file_type=file_type,
                purpose=purpose,
                key_components=[],
                imports=[],
                complexity_score=None,
                quality_notes=[f"Size: {file_meta.size} bytes"]
            )
            
            state['file_analyses'].append(file_analysis)
            print(f"  ✅ {file_meta.filename}")
            
        except Exception as e:
            state['error_messages'].append(f"File analysis error for {file_meta.filename}: {str(e)}")
    
    return state


def generate_insights(state: AnalysisState) -> AnalysisState:
    """Generate high-level project insights using Perplexity."""
    print("🧠 Generating project insights...")
    
    try:
        perplexity = PerplexityClient()
        
        # Prepare file summaries for analysis
        file_summaries = []
        for analysis in state['file_analyses']:
            final_project_summary = {
                'filename': Path(analysis.file_path).name,
                'type': analysis.file_type,
                'purpose': analysis.purpose,
                'components': analysis.key_components,
                'imports': analysis.imports
            }
            file_summaries.append(final_project_summary)
        
        # Get AI insights
        insights_text = perplexity.analyze_project_structure(file_summaries)
        
        # Estimate timeline based on file dates
        timeline = estimate_project_timeline(state['discovered_files'])
        
        # Extract research topics from imports and content
        research_topics = identify_research_topics(state['file_analyses'])
        frameworks = identify_frameworks(state['file_analyses'])
        
        insights = ProjectInsights(
            research_topics=research_topics,
            frameworks_used=frameworks,
            estimated_timeline=timeline,
            project_type="Detected from analysis",
            overall_quality=f"Based on {len(state['file_analyses'])} files - {insights_text[:200]}..."
        )
        
        state['project_insights'] = insights
        print("  ✅ Insights generated")
        
    except Exception as e:
        state['error_messages'].append(f"Insights generation error: {str(e)}")
    
    return state


def generate_summary(state: AnalysisState) -> AnalysisState:
    """Generate final project summary."""
    print("📋 Generating summary...")
    
    try:
        python_files = len([f for f in state['file_analyses'] if f.file_type == "Python"])
        total_files = len(state['file_analyses'])
        
        final_project_summary = f"""
# Project Analysis Summary

## Overview
- **Total Files**: {total_files} ({python_files} Python files)
- **Project Type**: {state['project_insights'].project_type}
- **Timeline**: {state['project_insights'].estimated_timeline}

## Research Focus
- **Topics**: {', '.join(state['project_insights'].research_topics)}
- **Frameworks**: {', '.join(state['project_insights'].frameworks_used)}

## File Analysis
"""
        
        for analysis in state['file_analyses'][:5]:  # Show first 5 files
            final_project_summary += f"### {Path(analysis.file_path).name}\n"
            final_project_summary += f"- **Type**: {analysis.file_type}\n"
            final_project_summary += f"- **Purpose**: {analysis.purpose[:100]}...\n"
            if analysis.key_components:
                final_project_summary += f"- **Components**: {', '.join(analysis.key_components[:3])}\n"
            final_project_summary += "\n"
        
        if len(state['file_analyses']) > 5:
            final_project_summary += f"... and {len(state['file_analyses']) - 5} more files\n\n"
        
        final_project_summary += f"## Quality Assessment\n{state['project_insights'].overall_quality}\n"
        
        if state['error_messages']:
            final_project_summary += f"\n## Errors Encountered\n"
            for error in state['error_messages'][:3]:
                final_project_summary += f"- {error}\n"
        
        state['analysis_report'] = final_project_summary
        print("  ✅ Summary generated")
        
    except Exception as e:
        state['error_messages'].append(f"Summary generation error: {str(e)}")
        state['analysis_report'] = "Error generating summary"
    
    return state


# Helper functions
def get_git_creation_date(file_path: Path) -> str:
    """Get the first Git commit date for a file."""
    try:
        cmd = f"git log --reverse --format=%ai --follow -- {file_path}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=file_path.parent)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except:
        pass
    return None


def extract_imports(tree: ast.AST) -> List[str]:
    """Extract import statements from AST."""
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend([alias.name for alias in node.names])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
    return list(set(imports))


def extract_components(tree: ast.AST) -> List[str]:
    """Extract classes and functions from AST."""
    components = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            components.append(f"Class: {node.name}")
        elif isinstance(node, ast.FunctionDef):
            components.append(f"Function: {node.name}")
    return components


def calculate_complexity(tree: ast.AST) -> int:
    """Calculate basic complexity score."""
    complexity = 1  # Base complexity
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.While, ast.For, ast.Try)):
            complexity += 1
    return complexity


def get_file_type_description(extension: str) -> str:
    """Get human-readable file type."""
    type_map = {
        '.html': 'HTML',
        '.css': 'CSS', 
        '.csv': 'CSV Data',
        '.parquet': 'Parquet Data',
        '.h5': 'HDF5 Data',
        '.hdf5': 'HDF5 Data',
        '.txt': 'Text'
    }
    return type_map.get(extension, 'Unknown')


def estimate_project_timeline(files: List[FileMetadata]) -> str:
    """Estimate project development timeline."""
    if not files:
        return "Unknown"
    
    # Find oldest and newest files
    dates = []
    for file in files:
        if file.git_first_commit:
            dates.append(file.git_first_commit)
        else:
            dates.append(file.creation_time)
    
    if dates:
        dates.sort()
        return f"Approximately {dates[0][:10]} to {dates[-1][:10]}"
    
    return "Timeline unknown"


def identify_research_topics(analyses: List[FileAnalysis]) -> List[str]:
    """Identify research topics from imports and content."""
    topics = []
    
    # Check imports for ML/AI libraries
    ml_keywords = {
        'torch': 'PyTorch/Deep Learning',
        'tensorflow': 'TensorFlow/Deep Learning', 
        'sklearn': 'Machine Learning',
        'transformers': 'Transformer Models',
        'pandas': 'Data Analysis',
        'numpy': 'Numerical Computing',
        'matplotlib': 'Data Visualization',
        'seaborn': 'Statistical Visualization'
    }
    
    for analysis in analyses:
        for import_name in analysis.imports:
            for keyword, topic in ml_keywords.items():
                if keyword in import_name.lower() and topic not in topics:
                    topics.append(topic)
    
    return topics


def identify_frameworks(analyses: List[FileAnalysis]) -> List[str]:
    """Identify frameworks used in the project."""
    frameworks = []
    
    framework_keywords = {
        'flask': 'Flask',
        'django': 'Django',
        'fastapi': 'FastAPI',
        'streamlit': 'Streamlit',
        'pytorch': 'PyTorch',
        'tensorflow': 'TensorFlow',
        'sklearn': 'Scikit-learn'
    }
    
    for analysis in analyses:
        for import_name in analysis.imports:
            for keyword, framework in framework_keywords.items():
                if keyword in import_name.lower() and framework not in frameworks:
                    frameworks.append(framework)
    
    return frameworks