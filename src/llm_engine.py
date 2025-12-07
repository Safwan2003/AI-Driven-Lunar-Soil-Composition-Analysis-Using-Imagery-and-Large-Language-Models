import os

class LLMEngine:
    def __init__(self, provider="mock", api_key=None):
        self.provider = provider
        self.api_key = api_key

    def generate_report(self, image_features, soil_composition):
        """
        Generates a scientific report based on extracted image features and composition data.
        
        Args:
            image_features (dict): Dictionary of detected terrain features (e.g., {'craters': True, 'regolith': 0.9})
            soil_composition (dict): Dictionary of elemental composition (e.g., {'Fe': 10, 'Ti': 5})
            
        Returns:
            str: The generated text report.
        """
        if self.provider == "mock":
            return self._mock_generation(image_features, soil_composition)
        elif self.provider == "openai":
            # Placeholder for OpenAI GPT-4V integration
            return "OpenAI integration not yet implemented."
        else:
            return "Unknown provider."

    def _mock_generation(self, image_features, soil_composition):
        """Simple template-based generation for testing without an API."""
        report = []
        report.append("## Automated Lunar Surface Analysis Report")
        report.append("\n**Observation Overview:**")
        
        features = [k for k, v in image_features.items() if v]
        if features:
            report.append(f"The analysis of the rover imagery identifies the following primary features: {', '.join(features)}.")
        else:
            report.append("No distinct terrain hazards identified.")
            
        report.append("\n**Soil Composition Estimates:**")
        elements = [f"- **{elem}**: {val}%" for elem, val in soil_composition.items()]
        report.append("\n".join(elements))
        
        report.append("\n**Scientific Interpretation:**")
        if soil_composition.get('Ti', 0) > 8:
            report.append("The high Titanium abundance suggests an Ilmenite-rich basalt, possibly from a young mare region.")
        else:
            report.append("The composition is consistent with standard lunar regolith samples.")
            
        return "\n".join(report)

if __name__ == "__main__":
    # Test the mock engine
    engine = LLMEngine()
    demo_features = {'Craters': True, 'Boulders': False}
    demo_composition = {'Fe': 12, 'Ti': 3, 'Si': 45}
    print(engine.generate_report(demo_features, demo_composition))
