"""
Prompt Manager Module
=====================

Manages prompts and templates for LLM interactions.
"""

from typing import Dict, Any, Optional


class PromptManager:
    """
    Manage and format prompts for LLM analysis.
    
    Provides a centralized system for prompt templates and formatting.
    """
    
    def __init__(self):
        """Initialize PromptManager with default templates."""
        self.templates = {
            "analysis": (
                "Analyze the following lunar soil composition data:\n\n"
                "{data}\n\n"
                "Provide detailed insights about the mineral composition, "
                "potential resources, and implications for lunar exploration."
            ),
            "classification": (
                "Based on the following classification results:\n\n"
                "{results}\n\n"
                "Explain the soil type, its characteristics, and significance."
            ),
            "comparison": (
                "Compare the following lunar soil samples:\n\n"
                "Sample A: {sample_a}\n"
                "Sample B: {sample_b}\n\n"
                "Highlight key differences and similarities."
            )
        }
    
    def get_prompt(
        self,
        template_name: str,
        variables: Dict[str, Any]
    ) -> str:
        """
        Get formatted prompt from template.
        
        Args:
            template_name: Name of the template to use
            variables: Variables to fill in the template
            
        Returns:
            Formatted prompt string
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        template = self.templates[template_name]
        return template.format(**variables)
    
    def add_template(self, name: str, template: str):
        """
        Add a new prompt template.
        
        Args:
            name: Name for the template
            template: Template string with placeholders
        """
        self.templates[name] = template
    
    def update_template(self, name: str, template: str):
        """
        Update an existing template.
        
        Args:
            name: Name of the template to update
            template: New template string
        """
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        self.templates[name] = template
