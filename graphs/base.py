"""
Base graph module for workflow definitions.

This module provides base classes for defining workflow graphs
used in the AI paper writing process.
"""

from typing import Dict, Any, List, Optional, Union, Callable, TypeVar, Generic
import json
import time
from pathlib import Path

from langchain.graphs import StateGraph
from langchain.schema.prompt_template import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from utils.logger import logger

# 제네릭 타입 변수 정의
T = TypeVar('T')

class BaseWorkflowGraph(Generic[T]):
    """
    Base class for workflow graphs used in the paper writing process.
    
    This class provides common functionality for creating and managing
    workflow graphs that orchestrate the paper writing process.
    """
    
    def __init__(self, name: str, description: str, state_class: type):
        """
        Initialize the workflow graph.
        
        Args:
            name (str): Name of the workflow graph
            description (str): Description of the workflow graph
            state_class (type): Class used for the workflow state
        """
        self.name = name
        self.description = description
        self.state_class = state_class
        self.graph = StateGraph(name=name)
        
        logger.info(f"Initialized BaseWorkflowGraph: {name}")
    
    def add_node(self, name: str, action: Callable[[T], T], description: str = "") -> None:
        """
        Add a node to the workflow graph.
        
        Args:
            name (str): Name of the node
            action (Callable): Action to be performed at this node
            description (str, optional): Description of the node
        """
        logger.info(f"Adding node to workflow: {name}")
        self.graph.add_node(name, action)
    
    def set_entry_point(self, node_name: str) -> None:
        """
        Set the entry point for the workflow.
        
        Args:
            node_name (str): Name of the entry point node
        """
        logger.info(f"Setting entry point to: {node_name}")
        self.graph.set_entry_point(node_name)
    
    def add_edge(self, start: str, end: str) -> None:
        """
        Add an edge between nodes in the workflow graph.
        
        Args:
            start (str): Name of the starting node
            end (str): Name of the ending node
        """
        logger.info(f"Adding edge to workflow: {start} -> {end}")
        self.graph.add_edge(start, end)
    
    def add_conditional_edges(self, start: str, condition: Callable[[T], str], possible_ends: List[str]) -> None:
        """
        Add conditional edges from a node.
        
        Args:
            start (str): Name of the starting node
            condition (Callable): Function that determines the next node
            possible_ends (List[str]): List of possible end nodes
        """
        logger.info(f"Adding conditional edges from: {start} to {possible_ends}")
        self.graph.add_conditional_edges(
            start,
            condition,
            {end: end for end in possible_ends}
        )
    
    def compile(self) -> None:
        """
        Compile the workflow graph.
        """
        logger.info("Compiling workflow graph")
        self.graph.compile()
    
    def run(self, state: T, config: Optional[RunnableConfig] = None) -> T:
        """
        Run the workflow with the given state.
        
        Args:
            state (T): Initial state for the workflow
            config (Optional[RunnableConfig]): Configuration for the runnable
            
        Returns:
            T: Final state from the workflow
        """
        logger.info(f"Running workflow: {self.name}")
        
        # Ensure the graph is compiled
        if not hasattr(self.graph, "_compiled") or not self.graph._compiled:
            logger.info("Graph not compiled, compiling now")
            self.compile()
        
        # Run the graph
        result = self.graph.invoke({"state": state}, config=config)
        final_state = result.get("state", state)
        
        logger.info(f"Workflow execution completed: {self.name}")
        return final_state
    
    def save_state(self, state: T, file_path: Optional[str] = None) -> str:
        """
        Save the workflow state to a file.
        
        Args:
            state (T): Workflow state to save
            file_path (Optional[str]): Path to save the state to
            
        Returns:
            str: Path where the state was saved
        """
        if file_path is None:
            # Create a timestamped filename
            timestamp = int(time.time())
            file_path = f"output/workflow_state_{timestamp}.json"
        
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert state to dict if it has a dict method
        if hasattr(state, "dict"):
            state_dict = state.dict()
        else:
            state_dict = state
        
        # Save to file
        with open(file_path, "w") as f:
            json.dump(state_dict, f, default=str, indent=2)
        
        logger.info(f"Saved workflow state to: {file_path}")
        return file_path
    
    def load_state(self, file_path: str) -> T:
        """
        Load workflow state from a file.
        
        Args:
            file_path (str): Path to load the state from
            
        Returns:
            T: Loaded workflow state
        """
        logger.info(f"Loading workflow state from: {file_path}")
        
        with open(file_path, "r") as f:
            state_dict = json.load(f)
        
        # Create a new state object
        state = self.state_class(**state_dict)
        
        return state
    
    def visualize(self, output_path: str = None) -> str:
        """
        Visualize the workflow graph.
        
        Args:
            output_path (Optional[str]): Path to save the visualization
            
        Returns:
            str: Path where the visualization was saved
        """
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("Visualization requires networkx and matplotlib. Install with: pip install networkx matplotlib")
            return ""
        
        if output_path is None:
            output_path = f"output/workflow_{self.name.replace(' ', '_').lower()}.png"
        
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for node in self.graph.nodes:
            G.add_node(node)
        
        # Add edges
        for edge in self.graph.edges:
            G.add_edge(edge[0], edge[1])
        
        # Draw the graph
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1500, font_size=10)
        
        # Add edge labels for conditional edges
        edge_labels = {}
        for edge in self.graph.edges:
            edge_labels[(edge[0], edge[1])] = ""
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        
        # Save the figure
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Saved workflow visualization to: {output_path}")
        return output_path


# BaseGraph as an alias for BaseWorkflowGraph for backward compatibility
BaseGraph = BaseWorkflowGraph