"""
Gemini API Client for LLM-Powered Lunar Analysis

Handles:
- Image + detection results → Natural language reports
- Chain-of-thought reasoning
- Scientific analysis generation
"""

import os
import base64
import json
from pathlib import Path
from typing import Dict, List, Optional
import google.generativeai as genai

class GeminiClient:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini API client.
        
        Args:
            api_key: Gemini API key. If None, reads from GEMINI_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter. Get your key at: https://ai.google.dev"
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def encode_image(self, image_path: str) -> str:
        """Convert image to base64 string."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def generate_terrain_report(
        self,
        image_path: str,
        terrain_class: str,
        confidence: float,
        composition: Dict[str, float]
    ) -> Dict[str, str]:
        """
        Generate comprehensive terrain analysis report.
        
        Args:
            image_path: Path to lunar surface image
            terrain_class: Detected terrain type (regolith/crater/boulder)
            confidence: Model confidence score (0-1)
            composition: Dict of elemental composition {element: percentage}
        
        Returns:
            Dict with: summary, analysis, recommendations, confidence_assessment
        """
        
        prompt = f"""You are an expert lunar geologist analyzing rover imagery for a space mission.

**Image Analysis Results:**
- Detected Terrain: {terrain_class}
- Model Confidence: {confidence:.1%}
- Estimated Composition:
{self._format_composition(composition)}

**Task:**
Provide a detailed scientific analysis in the following format:

1. **Summary** (2-3 sentences): What is visible in this image?

2. **Geological Analysis** (1 paragraph): 
   - Terrain characteristics
   - Formation process implications
   - Significance for lunar geology

3. **Composition Inference** (1 paragraph):
   - Analysis of detected elemental composition
   - Material properties and implications
   - Comparison to known lunar regions

4. **Mission Recommendations** (bullet points):
   - Scientific value for sampling
   - Rover traversability assessment
   - Safety considerations

5. **Confidence Assessment**: 
   - Evaluate the {confidence:.1%} confidence score
   - What factors support or challenge this classification?

Be specific, technical, and cite relevant lunar geology concepts. Keep total length under 400 words.
"""
        
        try:
            # For images, use the vision model
            response = self.model.generate_content(prompt)
            
            # Parse structured response
            full_text = response.text
            
            return {
                'full_report': full_text,
                'summary': self._extract_section(full_text, 'Summary'),
                'analysis': self._extract_section(full_text, 'Geological Analysis'),
                'composition_analysis': self._extract_section(full_text, 'Composition Inference'),
                'recommendations': self._extract_section(full_text, 'Mission Recommendations'),
                'confidence_assessment': self._extract_section(full_text, 'Confidence Assessment')
            }
        
        except Exception as e:
            return {
                'full_report': f"Error generating report: {str(e)}",
                'summary': "Report generation failed",
                'analysis': "",
                'composition_analysis': "",
                'recommendations': "",
                'confidence_assessment': ""
            }
    
    def generate_batch_summary(self, results: List[Dict]) -> str:
        """
        Generate summary across multiple image analyses.
        
        Args:
            results: List of analysis results from multiple images
        
        Returns:
            Cohesive summary report
        """
        prompt = f"""You are analyzing {len(results)} lunar surface images.

**Results Summary:**
{json.dumps([{
    'terrain': r['terrain_class'],
    'confidence': f"{r['confidence']:.0%}",
    'fe': f"{r.get('composition', {}).get('fe', 0):.1f}%"
} for r in results], indent=2)}

Provide a 200-word synthesis:
1. Overall terrain distribution
2. Compositional patterns
3. Recommendations for mission planning
"""
        
        response = self.model.generate_content(prompt)
        return response.text
    
    def _format_composition(self, comp: Dict[str, float]) -> str:
        """Format composition dict as readable string."""
        return '\n'.join([f"  - {k.upper()}: {v:.2f}%" for k, v in comp.items()])
    
    def _extract_section(self, text: str, section_name: str) -> str:
        """Extract a specific section from the LLM response."""
        try:
            # Find section by header
            lines = text.split('\n')
            section_lines = []
            in_section = False
            
            for line in lines:
                if section_name.lower() in line.lower() and any(marker in line for marker in ['**', '##', '###']):
                    in_section = True
                    continue
                elif in_section and any(marker in line for marker in ['**', '##', '###']) and len(section_lines) > 0:
                    break
                elif in_section:
                    section_lines.append(line)
            
            return '\n'.join(section_lines).strip()
        except:
            return ""

# Test function
if __name__ == "__main__":
    print("Testing Gemini Client...")
    print("NOTE: Requires GEMINI_API_KEY environment variable")
    
    try:
        client = GeminiClient()
        
        # Test with synthetic data
        report = client.generate_terrain_report(
            image_path="dummy.png",  # Not actually used in text-only test
            terrain_class="regolith",
            confidence=0.87,
            composition={
                'fe': 8.5,
                'mg': 4.2,
                'ti': 1.3,
                'si': 45.2
            }
        )
        
        print("\n" + "=" * 60)
        print("GENERATED REPORT:")
        print("=" * 60)
        print(report['full_report'])
        
    except ValueError as e:
        print(f"[ERROR] {e}")
        print("\nTo test, set your API key:")
        print("  Windows: $env:GEMINI_API_KEY='your-key-here'")
        print("  Linux/Mac: export GEMINI_API_KEY='your-key-here'")
