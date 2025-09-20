"""Processing nodes for the project analysis agent."""

import os
import ast
import subprocess
import math
import re
from datetime import datetime
from pathlib import Path
from typing import List
from difflib import SequenceMatcher  # ← This is built-in, no pip install needed

from state import AnalysisState, FileMetadata, FileAnalysis, ProjectInsights
from perplexity_client import PerplexityClient


def discover_files(state: AnalysisState) -> AnalysisState:
    """Discover and catalog all relevant files in the directory."""
    print(f"🔍 Scanning directory: {state['target_directory']}")
    
    target_extensions = {'.py', '.html', '.css'} #, '.csv', '.parquet', '.h5', '.hdf5', '.txt'}
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


def calculate_maintainability_index(tree: ast.AST, file_content: str) -> int:
    """
    Calculate Maintainability Index (0-100 scale).
    Formula: MI = 171 - 5.2*ln(V) - 0.23*G - 16.2*ln(L)
    """
    import math
    
    lines_of_code = len([line for line in file_content.split('\n') if line.strip()])
    cyclomatic_complexity = calculate_complexity(tree)
    
    # FIXED: Use the correct function name and extract volume
    halstead_metrics = calculate_halstead_metrics(tree)
    halstead_volume = halstead_metrics['volume']
    
    # Avoid log(0) errors
    if halstead_volume <= 0 or lines_of_code <= 0:
        return 50  # Neutral score
    
    mi = 171 - 5.2 * math.log(halstead_volume) - 0.23 * cyclomatic_complexity - 16.2 * math.log(lines_of_code)
    
    # Normalize to 0-100 scale
    return max(0, min(100, int(mi)))


def generate_summary(state: AnalysisState) -> AnalysisState:
    """Generate comprehensive markdown report with all metrics."""
    print("📋 Generating enhanced summary...")
    
    try:
        python_files = len([f for f in state['file_analyses'] if f.file_type == "Python"])
        total_files = len(state['file_analyses'])
        
        md_parts = [
            "# 📊 Project Analysis Summary",
            "",
            "## Overview",
            f"- **Total Files**: {total_files} ({python_files} Python files)",
            f"- **Project Type**: {state['project_insights'].project_type}",
            f"- **Timeline**: {state['project_insights'].estimated_timeline}",
            "",
            "## Research Focus",
            f"- **Topics**: {', '.join(state['project_insights'].research_topics) or 'None identified'}",
            f"- **Frameworks**: {', '.join(state['project_insights'].frameworks_used) or 'None identified'}",
            ""
        ]
        
        # FIXED: Safe access to duplication analysis
        dup_data = state.get('duplication_analysis')  # ← This won't crash if key doesn't exist
        if dup_data:
            md_parts.extend([
                "## Code Duplication Analysis",
                f"- **Total Duplications Found**: {dup_data['total_duplications']}",
                f"- **Duplication Percentage**: {dup_data['duplication_percentage']}%",
                ""
            ])
            
            if dup_data['duplications']:
                md_parts.append("### Detected Duplications:")
                for dup in dup_data['duplications']:
                    md_parts.append(f"- **{dup['file1']}** ↔ **{dup['file2']}**: {dup['similarity']}% similarity")
                md_parts.append("")
        
        # Rest of the function remains the same...
        md_parts.extend([
            "## Detailed File Analysis",
            ""
        ])
        
        for analysis in state['file_analyses']:
            md_parts.extend([
                f"### `{Path(analysis.file_path).name}`",
                "",
                f"**File Type**: {analysis.file_type}",
                "",
                "**Purpose**:",
                f"> {analysis.purpose}",
                "",
                f"**Key Components**: {', '.join(analysis.key_components) or '—'}",
                f"**Imports**: {', '.join(analysis.imports) or '—'}",
                ""
            ])
            
            if analysis.quality_notes:
                md_parts.extend([
                    "**Quality Metrics**:",
                    ""
                ])
                for note in analysis.quality_notes:
                    if note.startswith("⚠️"):
                        md_parts.append(f"- 🚨 {note}")
                    else:
                        md_parts.append(f"- {note}")
                md_parts.append("")
        
        # Overall assessment
        md_parts.extend([
            "## Quality Assessment",
            "",
            state['project_insights'].overall_quality,
            ""
        ])
        
        if state['error_messages']:
            md_parts.extend([
                "## Issues Encountered",
                ""
            ])
            for error in state['error_messages']:
                md_parts.append(f"- ⚠️ {error}")
        
        state['analysis_report'] = "\n".join(md_parts)
        print("  ✅ Enhanced summary generated")
        
    except Exception as e:
        state['error_messages'].append(f"Summary generation error: {str(e)}")
        state['analysis_report'] = f"# Error Generating Summary\n\n{str(e)}"
    
    return state



def calculate_halstead_metrics(tree: ast.AST) -> dict:
    """Calculate Halstead complexity metrics from AST."""
    operators = set()
    operands = set()
    total_operators = 0
    total_operands = 0
    
    for node in ast.walk(tree):
        # Count operators (control structures, assignments, etc.)
        if isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, 
                           ast.Pow, ast.LShift, ast.RShift, ast.BitOr, 
                           ast.BitXor, ast.BitAnd, ast.FloorDiv)):
            operators.add(type(node).__name__)
            total_operators += 1
        elif isinstance(node, (ast.And, ast.Or, ast.Not, ast.Invert, 
                             ast.UAdd, ast.USub)):
            operators.add(type(node).__name__)
            total_operators += 1
        elif isinstance(node, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, 
                             ast.Gt, ast.GtE, ast.Is, ast.IsNot, 
                             ast.In, ast.NotIn)):
            operators.add(type(node).__name__)
            total_operators += 1
        
        # Count operands (variables, constants, function names)
        elif isinstance(node, ast.Name):
            operands.add(node.id)
            total_operands += 1
        elif isinstance(node, (ast.Constant, ast.Num, ast.Str)):
            operands.add(str(getattr(node, 'n', getattr(node, 's', node.value))))
            total_operands += 1
    
    # Halstead metrics
    n1 = len(operators)    # unique operators
    n2 = len(operands)     # unique operands
    N1 = total_operators   # total operators
    N2 = total_operands    # total operands
    
    vocabulary = n1 + n2
    length = N1 + N2
    volume = length * (vocabulary.bit_length() if vocabulary > 0 else 1)
    difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
    effort = difficulty * volume
    
    return {
        'vocabulary': vocabulary,
        'length': length,
        'volume': volume,
        'difficulty': round(difficulty, 2),
        'effort': round(effort, 2),
        'time_to_program': round(effort / 18, 2)  # seconds
    }


def detect_code_duplication(file_analyses: List[FileAnalysis]) -> dict:
    """Detect potential code duplication across files."""
    from difflib import SequenceMatcher
    
    duplications = []
    similarity_threshold = 0.7
    
    python_files = [f for f in file_analyses if f.file_type == "Python"]
    
    for i, file1 in enumerate(python_files):
        for j, file2 in enumerate(python_files[i+1:], i+1):
            # Compare function signatures
            common_functions = set(file1.key_components) & set(file2.key_components)
            if common_functions:
                similarity = len(common_functions) / max(len(file1.key_components), len(file2.key_components))
                if similarity > similarity_threshold:
                    duplications.append({
                        'file1': Path(file1.file_path).name,
                        'file2': Path(file2.file_path).name,
                        'similarity': round(similarity * 100, 1),
                        'common_elements': list(common_functions)
                    })
    
    return {
        'total_duplications': len(duplications),
        'duplications': duplications,
        'duplication_percentage': round(len(duplications) / len(python_files) * 100, 1) if python_files else 0
    }



def analyze_dependency_complexity(tree: ast.AST) -> dict:
    """Analyze import dependencies and coupling."""
    imports = []
    from_imports = []
    local_imports = 0
    external_imports = 0
    
    standard_libs = {'os', 'sys', 'json', 'datetime', 'math', 're', 'collections', 
                     'itertools', 'functools', 'pathlib', 'typing', 'ast', 'subprocess'}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
                if alias.name.split('.')[0] in standard_libs:
                    external_imports += 1
                else:
                    local_imports += 1
                    
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                from_imports.append(node.module)
                if node.module.split('.')[0] in standard_libs:
                    external_imports += 1
                else:
                    local_imports += 1
    
    return {
        'total_imports': len(imports) + len(from_imports),
        'standard_library_imports': external_imports,
        'local_imports': local_imports,
        'coupling_ratio': round(local_imports / (local_imports + external_imports) * 100, 1) if (local_imports + external_imports) > 0 else 0,
        'import_diversity': len(set(imports + from_imports))
    }



def analyze_security_patterns(tree: ast.AST, file_content: str) -> dict:
    """Detect basic security anti-patterns."""
    security_issues = []
    
    # Check for hardcoded secrets
    import re
    secret_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][^"\']+["\']'
    ]
    
    for pattern in secret_patterns:
        if re.search(pattern, file_content, re.IGNORECASE):
            security_issues.append("Potential hardcoded secret")
    
    # Check for dangerous functions
    dangerous_calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id in ['eval', 'exec', 'compile']:
                    dangerous_calls.append(f"Use of {node.func.id}")
    
    return {
        'security_score': max(0, 100 - len(security_issues) * 20 - len(dangerous_calls) * 30),
        'issues': security_issues + dangerous_calls,
        'total_issues': len(security_issues) + len(dangerous_calls)
    }


def analyze_python_files(state: AnalysisState) -> AnalysisState:
    """Enhanced Python file analysis with comprehensive metrics."""
    print("🐍 Analyzing Python files with enhanced metrics...")
    
    perplexity = PerplexityClient()
    python_files = [f for f in state['discovered_files'] if f.extension == '.py']
    
    for file_meta in python_files:
        try:
            with open(file_meta.path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # AST analysis
            tree = ast.parse(content)
            
            # Basic metrics
            imports = extract_imports(tree)
            components = extract_components(tree)
            complexity = calculate_complexity(tree)
            
            # Enhanced metrics
            maintainability = calculate_maintainability_index(tree, content)
            halstead = calculate_halstead_metrics(tree)
            dependency_analysis = analyze_dependency_complexity(tree)
            security_analysis = analyze_security_patterns(tree, content)
            
            # AI analysis
            analysis_result = perplexity.analyze_code(content, "python")
            
            # Create comprehensive quality notes
            quality_notes = [
                f"Complexity: {complexity}",
                f"Maintainability Index: {maintainability}/100",
                f"Halstead Volume: {halstead['volume']}",
                f"Import Coupling: {dependency_analysis['coupling_ratio']}%",
                f"Security Score: {security_analysis['security_score']}/100"
            ]
            
            if security_analysis['issues']:
                quality_notes.extend([f"⚠️ {issue}" for issue in security_analysis['issues']])
            
            file_analysis = FileAnalysis(
                file_path=file_meta.path,
                file_type="Python",
                purpose=f"AST Analysis + AI: {analysis_result}",
                key_components=components,
                imports=imports,
                complexity_score=complexity,
                quality_notes=quality_notes
            )
            
            state['file_analyses'].append(file_analysis)
            print(f"  ✅ {file_meta.filename} (MI: {maintainability}, Security: {security_analysis['security_score']})")
            
        except Exception as e:
            state['error_messages'].append(f"Enhanced analysis error for {file_meta.filename}: {str(e)}")
    
    # Project-level duplication analysis
    if len(state['file_analyses']) > 1:
        duplication_report = detect_code_duplication(state['file_analyses'])
        state['project_insights'].duplication_analysis = duplication_report
    
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
                    purpose += f" | Sample: {sample}..."
            
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
            file_summary = {
                'filename': Path(analysis.file_path).name,
                'type': analysis.file_type,
                'purpose': analysis.purpose,
                'components': analysis.key_components,
                'imports': analysis.imports
            }
            file_summaries.append(file_summary)
        
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
    """Generate comprehensive markdown report with all metrics."""
    print("📋 Generating enhanced summary...")
    
    try:
        python_files = len([f for f in state['file_analyses'] if f.file_type == "Python"])
        total_files = len(state['file_analyses'])
        
        # Build comprehensive markdown report
        md_parts = [
            "# 📊 Project Analysis Summary",
            "",
            "## Overview",
            f"- **Total Files**: {total_files} ({python_files} Python files)",
            f"- **Project Type**: {state['project_insights'].project_type}",
            f"- **Timeline**: {state['project_insights'].estimated_timeline}",
            "",
            "## Research Focus",
            f"- **Topics**: {', '.join(state['project_insights'].research_topics) or 'None identified'}",
            f"- **Frameworks**: {', '.join(state['project_insights'].frameworks_used) or 'None identified'}",
            ""
        ]
        
        # Add duplication analysis if available
        if state.get('duplication_analysis'):
            dup_data = state.get('duplication_analysis')
            md_parts.extend([
                "## Code Duplication Analysis",
                f"- **Total Duplications Found**: {dup_data['total_duplications']}",
                f"- **Duplication Percentage**: {dup_data['duplication_percentage']}%",
                ""
            ])
            
            if dup_data['duplications']:
                md_parts.append("### Detected Duplications:")
                for dup in dup_data['duplications']:
                    md_parts.append(f"- **{dup['file1']}** ↔ **{dup['file2']}**: {dup['similarity']}% similarity")
                md_parts.append("")
        
        # Detailed file analysis
        md_parts.extend([
            "## Detailed File Analysis",
            ""
        ])
        
        for analysis in state['file_analyses']:
            md_parts.extend([
                f"### `{Path(analysis.file_path).name}`",
                "",
                f"**File Type**: {analysis.file_type}  ",
                f"**Size**: {analysis.quality_notes[-1] if 'Size:' in str(analysis.quality_notes) else 'Unknown'}",
                "",
                "**Purpose**:",
                f"> {analysis.purpose}",
                "",
                f"**Key Components**: {', '.join(analysis.key_components) or '—'}  ",
                f"**Imports**: {', '.join(analysis.imports) or '—'}  ",
                ""
            ])
            
            # Enhanced quality metrics (if available)
            if analysis.quality_notes:
                md_parts.extend([
                    "**Quality Metrics**:",
                    ""
                ])
                for note in analysis.quality_notes:
                    if note.startswith("⚠️"):
                        md_parts.append(f"- 🚨 {note}")
                    else:
                        md_parts.append(f"- {note}")
                md_parts.append("")
        
        # Overall assessment
        md_parts.extend([
            "## Quality Assessment",
            "",
            state['project_insights'].overall_quality,
            ""
        ])
        
        # Errors section (if any)
        if state['error_messages']:
            md_parts.extend([
                "## Issues Encountered",
                ""
            ])
            for error in state['error_messages']:
                md_parts.append(f"- ⚠️ {error}")
            md_parts.append("")
        
        # Final recommendations
        md_parts.extend([
            "## Recommendations",
            "",
            "### Code Quality Improvements",
            "- Files with complexity > 10 should be refactored",
            "- Files with maintainability index < 50 need attention", 
            "- Address any security issues flagged above",
            "",
            "### Next Steps",
            "- Review files with low quality scores",
            "- Consider adding unit tests for complex functions",
            "- Document any missing docstrings",
            "",
            f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        state['analysis_report'] = "\n".join(md_parts)
        print("  ✅ Enhanced summary generated")
        
    except Exception as e:
        state['error_messages'].append(f"Summary generation error: {str(e)}")
        state['analysis_report'] = f"# Error Generating Summary\n\n{str(e)}"
    
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