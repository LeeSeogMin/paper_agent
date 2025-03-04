"""
Base graph module for workflow definitions.

This module provides base classes for defining workflow graphs
used in the AI paper writing process.
"""

from typing import Dict, Any, List, Optional, Union, Callable
import json

from langchain.graphs import StateGraph
from langchain.schema.prompt_template import PromptTemplate
from langchain.chat_models import ChatOpenAI

from utils.logger import logger


class BaseWorkflowGraph:
    """
    Base class for workflow graphs used in the paper writing process.
    
    This class provides common functionality for creating and managing
    workflow graphs that orchestrate the paper writing process.
    """
    
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.2):
        """
        Initialize the workflow graph.
        
        Args:
            model_name (str): Name of the language model to use
            temperature (float): Temperature setting for the language model
        """
        self.model_name = model_name
        self.temperature = temperature
        self.graph = None
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        
        logger.info(f"Initialized BaseWorkflowGraph with model: {model_name}")
    
    def create_graph(self) -> StateGraph:
        """
        Create the workflow graph structure.
        
        Returns:
            StateGraph: The created workflow graph
        """
        # This method should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement create_graph()")
    
    def add_node(self, name: str, action: Any, description: str) -> None:
        """
        Add a node to the workflow graph.
        
        Args:
            name (str): Name of the node
            action (Any): Action to be performed at this node
            description (str): Description of the node
        """
        if self.graph is None:
            raise ValueError("Graph has not been created. Call create_graph() first.")
        
        logger.info(f"Adding node to workflow: {name}")
        self.graph.add_node(name, action, description=description)
    
    def add_edge(self, start: str, end: str, condition: Optional[Callable] = None) -> None:
        """
        Add an edge between nodes in the workflow graph.
        
        Args:
            start (str): Name of the starting node
            end (str): Name of the ending node
            condition (Optional[Callable]): Optional condition for traversing the edge
        """
        if self.graph is None:
            raise ValueError("Graph has not been created. Call create_graph() first.")
        
        logger.info(f"Adding edge to workflow: {start} -> {end}")
        if condition:
            self.graph.add_conditional_edges(start, condition, {True: end, False: end})
        else:
            self.graph.add_edge(start, end)
    
    def compile(self) -> None:
        """
        Compile the workflow graph.
        """
        if self.graph is None:
            raise ValueError("Graph has not been created. Call create_graph() first.")
        
        logger.info("Compiling workflow graph")
        self.graph.compile()
    
    def run_workflow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the workflow with the given inputs.
        
        Args:
            inputs (Dict[str, Any]): Input data for the workflow
            
        Returns:
            Dict[str, Any]: Output data from the workflow
        """
        if self.graph is None:
            raise ValueError("Graph has not been created. Call create_graph() first.")
        
        logger.info(f"Running workflow with inputs: {json.dumps(inputs, default=str)[:200]}...")
        
        # Check if the graph is compiled
        if not hasattr(self.graph, "_compiled") or not self.graph._compiled:
            logger.info("Graph not compiled, compiling now")
            self.compile()
        
        # Run the graph
        result = self.graph.invoke(inputs)
        
        logger.info("Workflow execution completed")
        return result