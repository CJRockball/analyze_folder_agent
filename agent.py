"""LangGraph agent implementation for project analysis."""

from langgraph.graph import StateGraph, END
from state import AnalysisState, ProjectInsights
from nodes import (
    discover_files,
    analyze_python_files, 
    analyze_other_files,
    generate_insights,
    generate_summary
)


def create_analysis_agent() -> StateGraph:
    """Create the LangGraph workflow for project analysis."""
    
    # Define the workflow graph
    workflow = StateGraph(AnalysisState)
    
    # Add nodes
    workflow.add_node("discover", discover_files)
    workflow.add_node("analyze_python", analyze_python_files)
    workflow.add_node("analyze_other", analyze_other_files)  
    workflow.add_node("insights", generate_insights)
    workflow.add_node("summary_generator", generate_summary)
    
    # Define the flow
    workflow.set_entry_point("discover")
    workflow.add_edge("discover", "analyze_python")
    workflow.add_edge("analyze_python", "analyze_other")
    workflow.add_edge("analyze_other", "insights")
    workflow.add_edge("insights", "summary_generator")
    workflow.add_edge("summary_generator", END)
    
    return workflow.compile()


def run_analysis(target_directory: str) -> dict:
    """Run the complete project analysis."""
    
    # Initialize state
    initial_state = {
        "target_directory": target_directory,
        "discovered_files": [],
        "file_analyses": [],
        "project_insights": ProjectInsights(
            research_topics=[],
            frameworks_used=[],
            estimated_timeline="Unknown",
            project_type="Unknown",
            overall_quality="Not assessed"
        ),
        "analysis_report": "",
        "error_messages": []
    }
    
    # Create and run the agent
    agent = create_analysis_agent()
    result = agent.invoke(initial_state)
    
    return result