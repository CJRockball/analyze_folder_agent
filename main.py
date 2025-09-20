"""Main entry point for the project analysis agent."""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

from agent import run_analysis


def main():
    """Main CLI function."""
    # Load environment variables
    load_dotenv()
    
    # Check for target directory argument
    if len(sys.argv) != 2:
        print("Usage: python main.py <target_directory>")
        sys.exit(1)
    
    target_directory = sys.argv[1]
    
    # Validate directory exists
    if not Path(target_directory).exists():
        print(f"Error: Directory '{target_directory}' does not exist")
        sys.exit(1)
    
    # Check for Perplexity API key
    if not os.getenv("PERPLEXITY_API_KEY"):
        print("Error: PERPLEXITY_API_KEY environment variable not set")
        print("Set it with: export PERPLEXITY_API_KEY='your-api-key'")
        sys.exit(1)
    
    print("🚀 Starting project analysis...")
    print(f"📂 Target directory: {target_directory}")
    print("-" * 50)
    
    # In main.py, update the file saving section:
    try:
        # Run the analysis
        result = run_analysis(target_directory)
        
        # Display results
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(result['analysis_report'])
        
        # Save comprehensive results to markdown file
        output_file = Path(target_directory) / "analysis_results.md"  # Changed to .md
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result['analysis_report'])
        
        # Also save raw data as JSON for further processing
        json_file = Path(target_directory) / "analysis_data.json"
        import json
        raw_data = {
            'file_analyses': [analysis.__dict__ if hasattr(analysis, '__dict__') else analysis for analysis in result['file_analyses']],
            'project_insights': result['project_insights'].__dict__ if hasattr(result['project_insights'], '__dict__') else result['project_insights'],
            'duplication_analysis': result.get('duplication_analysis', {}),
            'error_messages': result['error_messages']
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, default=str)
        
        print(f"\n💾 Markdown report saved to: {output_file}")
        print(f"💾 Raw data saved to: {json_file}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()