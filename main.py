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
    
    try:
        # Run the analysis
        result = run_analysis(target_directory)
        
        # Display results
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(result['analysis_report'])
        
        # Save detailed results
        output_file = Path(target_directory) / "analysis_results.txt"
        with open(output_file, 'w') as f:
            f.write(result['analysis_report'])
            f.write("\n\n=== DETAILED FILE ANALYSES ===\n")
            for analysis in result['file_analyses']:
                f.write(f"\n## {Path(analysis.file_path).name}\n")
                f.write(f"Purpose: {analysis.purpose}\n")
                f.write(f"Components: {', '.join(analysis.key_components)}\n")
                f.write(f"Imports: {', '.join(analysis.imports)}\n")
                f.write(f"Quality Notes: {', '.join(analysis.quality_notes)}\n")
        
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        if result['error_messages']:
            print(f"\n⚠️  {len(result['error_messages'])} errors encountered during analysis")
            for error in result['error_messages'][:3]:
                print(f"   - {error}")
    
    except Exception as e:
        print(f"\n❌ Analysis failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()