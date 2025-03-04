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
    
    # Add new argument for user requirements
    parser.add_argument("--requirements", type=str, default=None, 
                       help="Detailed requirements for the paper in free text format")
    
    # Make topic optional since it might be extracted from requirements
    parser.add_argument("--topic", type=str, required=False, help="Paper topic")
    
    # Since DEFAULT_TEMPLATE and other constants are commented out, we handle default values as strings
    parser.add_argument("--template", type=str, default="academic", help="Paper template name")
    parser.add_argument("--style", type=str, default="Standard Academic", help="Style guide name") 
    parser.add_argument("--citation", type=str, default="APA", help="Citation style")
    parser.add_argument("--format", type=str, default="markdown", choices=["markdown", "latex"], help="Output format")
    parser.add_argument("--output", type=str, default=None, help="Output file path (default: auto-generated)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--log-level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    
    return parser.parse_args()


def process_requirements(requirements: str) -> Dict[str, Any]:
    """
    Process user requirements using LLM to extract structured information
    
    Args:
        requirements (str): Free-form text containing user requirements
        
    Returns:
        Dict[str, Any]: Extracted parameters including topic, style, etc.
    """
    try:
        # Here you would integrate with your LLM to analyze the requirements
        # For now, we'll return a basic structure
        # TODO: Implement LLM integration
        return {
            "topic": None,
            "template": "academic",
            "style": "Standard Academic",
            "citation": "APA",
            "format": "markdown",
            "additional_requirements": requirements
        }
    except Exception as e:
        logger.error(f"Error processing requirements: {str(e)}")
        return {}


def main():
    """
    Main application function
    """
    args = parse_arguments()
    
    # Configure logging
    configure_logging(args.log_level)
    
    # Get requirements from user if not provided via command line
    requirements = args.requirements
    if not requirements:
        print("\nPlease describe your paper requirements in detail.")
        print("You can include topic, style preferences, specific sections needed, etc.")
        print("Example: 'I need a technical paper about Graph Neural Networks focusing on")
        print("recent applications in social networks, using IEEE format with LaTeX output.'")
        requirements = input("\nYour requirements: ").strip()
    
    # Process requirements using LLM
    if requirements:
        params = process_requirements(requirements)
        
        # Update args with extracted parameters if not explicitly provided
        if not args.topic and params.get("topic"):
            args.topic = params["topic"]
        if not args.template and params.get("template"):
            args.template = params["template"]
        if not args.style and params.get("style"):
            args.style = params["style"]
        if not args.citation and params.get("citation"):
            args.citation = params["citation"]
        if not args.format and params.get("format"):
            args.format = params["format"]
    
    # Request topic from user if still not available
    if not args.topic:
        args.topic = input("Please enter the paper topic: ")
    
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