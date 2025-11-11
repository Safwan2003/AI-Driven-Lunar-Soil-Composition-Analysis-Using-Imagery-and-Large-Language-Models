# src/llm_integration/interpretation.py

"""
This script integrates the vision model's output with a Large Language Model (LLM)
to generate explainable, natural-language reports.

Functions:
- generate_report: Combines classification results with LLM reasoning.
- get_llm_explanation: Queries an LLM for an explanation of the findings.
"""

import openai

def generate_report(classification_results, mineral_data):
    """
    Generates a natural-language report from model outputs.
    
    Args:
        classification_results: The output from the vision model.
        mineral_data: The detected mineral composition.
        
    Returns:
        A string containing the generated report.
    """
    explanation = get_llm_explanation(classification_results, mineral_data)
    report = f"""
    **Lunar Soil Analysis Report**

    **Classification:** {classification_results}
    **Mineralogy:** {mineral_data}

    **Interpretation:**
    {explanation}
    """
    return report

def get_llm_explanation(classification, minerals):
    """

    Queries an LLM to get a human-readable explanation of the analysis.
    
    Args:
        classification: The soil type classification.
        minerals: A dictionary of detected minerals and their abundance.
        
    Returns:
        A string with the LLM's explanation.
    """
    # This requires an API key for OpenAI
    # openai.api_key = "YOUR_API_KEY"
    
    prompt = f"Explain the significance of a lunar soil sample classified as '{classification}' with the following mineral composition: {minerals}. Focus on its potential for resource utilization."
    
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Could not generate explanation: {e}"
