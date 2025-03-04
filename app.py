# Required module imports
import os
import sys
from dotenv import load_dotenv

# Set project root path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Load environment variables
load_dotenv()

# (temporary) Agent imports commented out
# from agents.research_agent import ResearchAgent
# from agents.writing_agent import WritingAgent
# from agents.review_agent import ReviewAgent

import argparse
import json
from typing import Dict, Any, List, Optional, Union

# (temporary) Settings imports commented out
# from config.settings import OPENAI_MODEL, DEFAULT_TEMPLATE, OUTPUT_DIR
from utils.logger import logger, configure_logging
from models.state import PaperWorkflowState
from graphs.paper_writing import PaperWritingGraph


def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="AI Paper Writing Agent")
    
    # Since DEFAULT_TEMPLATE and other constants are commented out, we handle default values as strings
    parser.add_argument("--topic", type=str, required=True, help="Paper topic")
    parser.add_argument("--template", type=str, default="academic", help="Paper template name")
    parser.add_argument("--style", type=str, default="Standard Academic", help="Style guide name") 
    parser.add_argument("--citation", type=str, default="APA", help="Citation style")
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "latex"], help="Output format")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: auto-generated)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    return parser.parse_args()


def main():
    """
    Main application function
    """
    args = parse_arguments()
    
    # Configure logging
    configure_logging(args.log_level)
    
    # Request topic from user if not provided
    if not args.topic:
        args.topic = input("Graph Neural Network Topic Modeling: ")
    
    # If template not provided
    if not args.template:
        args.template = input("Enter paper template (default: academic): ") or "academic"
    
    # (temporary) Set OUTPUT_DIR manually (config.settings import is commented out)
    OUTPUT_DIR = os.path.join(BASE_DIR, "output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Create paper writing graph
        paper_graph = PaperWritingGraph()
        
        # Run workflow
        final_state = paper_graph.run_workflow(
            topic=args.topic,
            template_name=args.template,
            style_guide=args.style,
            citation_style=args.citation,
            output_format=args.format,
            verbose=args.verbose
        )
        
        # Output results
        if final_state.error:
            logger.error(f"Workflow error: {final_state.error}")
            print(f"Error: {final_state.error}")
            return 1
        
        # Successful execution
        print("\n" + "=" * 50)
        print(f"Paper writing completed: {final_state.paper.title}")
        print(f"Number of sections: {len(final_state.paper.sections)}")
        print(f"Number of references: {len(final_state.paper.references)}")
        
        if final_state.output_file:
            print(f"Output file: {final_state.output_file}")
        
        # Output review information
        if final_state.review:
            print("\n[Paper Review]")
            review = final_state.review
            print(f"Overall rating: {review.get('overall_rating', 'N/A')}/10")
            
            if 'strengths' in review and review['strengths']:
                print("\nStrengths:")
                for strength in review['strengths'][:3]:
                    print(f"- {strength}")
            
            if 'suggestions' in review and review['suggestions']:
                print("\nImprovement suggestions:")
                for suggestion in review['suggestions'][:3]:
                    print(f"- {suggestion}")
        
        print("=" * 50 + "\n")
        return 0
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())