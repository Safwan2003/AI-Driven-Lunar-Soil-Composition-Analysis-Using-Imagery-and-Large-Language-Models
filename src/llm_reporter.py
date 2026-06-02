"""
LLM-powered scientific report generator for SUPARCO Soil Analysis.
Uses Google Gemini (new google.genai SDK) to produce soil science narratives.
"""

import os
import io
import logging
from pathlib import Path
from typing import Optional, Dict
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

SYSTEM_CONTEXT = (
    "You are a senior soil scientist and environmental chemist working with SUPARCO, "
    "Pakistan's national space agency. You analyse heavy metal concentrations in soil "
    "samples captured via AI-driven imagery. Write scientifically rigorous, concise "
    "reports that are accessible to both scientists and mission planners."
)

# Reference thresholds (mg/kg) — WHO/FAO soil guidelines + literature
SAFE_THRESHOLDS = {
    'Cd': 0.3,   # Cadmium — WHO guideline: 0.3 mg/kg agricultural soil
    'Cu': 36.0,  # Copper — typical background ~20 mg/kg
    'Ni': 35.0,  # Nickel — typical background ~20–35 mg/kg
    'Mn': 400.0, # Manganese — typical 200–400 mg/kg
    'Fe': 40000.0,  # Iron — typical crustal ~40,000 mg/kg (4%)
    'Zn': 70.0,  # Zinc — typical background ~50–70 mg/kg
}

ELEMENT_FULL = {
    'Cd': 'Cadmium', 'Cu': 'Copper', 'Ni': 'Nickel',
    'Mn': 'Manganese', 'Fe': 'Iron', 'Zn': 'Zinc'
}


def _load_api_key():
    key = os.environ.get('GEMINI_API_KEY')
    if key:
        return key
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith('GEMINI_API_KEY='):
                return line.split('=', 1)[1].strip().strip('"\'')
    return None


class SoilLLMReporter:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _load_api_key()
        self._client = None
        self._model_name = 'gemini-2.0-flash'
        self._initialized = False
        if self.api_key:
            self._init()

    def _init(self):
        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            self._initialized = True
            logger.info("Gemini client initialised (google.genai SDK)")
        except Exception as e:
            logger.error(f"Gemini init failed: {e}")

    @property
    def is_available(self):
        return self._initialized and self._client is not None

    def generate_report(
        self,
        composition: Dict[str, float],
        terrain_class: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        model_metrics: Optional[Dict] = None,
    ) -> str:
        """
        Generate a scientific soil analysis report.

        Args:
            composition: {element: predicted_value_mg_kg}
            terrain_class: optional terrain label
            image: optional RGB numpy image for visual context
            model_metrics: optional dict of MAE/R² values to mention in report
        """
        prompt = self._build_prompt(composition, terrain_class, model_metrics)

        if self.is_available:
            try:
                if image is not None:
                    return self._call_with_image(prompt, image)
                return self._call_text(prompt)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")

        return self._fallback_report(composition, terrain_class)

    def _call_text(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=SYSTEM_CONTEXT + "\n\n" + prompt,
        )
        return response.text

    def _call_with_image(self, prompt: str, image: np.ndarray) -> str:
        from google.genai import types
        pil_img = Image.fromarray(image.astype(np.uint8))
        if max(pil_img.size) > 1024:
            pil_img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=85)
        img_part = types.Part.from_bytes(data=buf.getvalue(), mime_type='image/jpeg')
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=[SYSTEM_CONTEXT + "\n\n" + prompt, img_part],
        )
        return response.text

    def _build_prompt(self, comp: Dict, terrain: Optional[str], metrics: Optional[Dict]) -> str:
        lines = []
        lines.append("## Soil Sample Analysis Data\n")
        lines.append("### Predicted Heavy Metal Concentrations (mg/kg)\n")
        for e, v in comp.items():
            flag = " ⚠️ ABOVE THRESHOLD" if v > SAFE_THRESHOLDS.get(e, 1e9) else ""
            lines.append(f"- {ELEMENT_FULL[e]} ({e}): **{v:.3f} mg/kg** | Safe limit: {SAFE_THRESHOLDS.get(e,'N/A')}{flag}")

        if terrain:
            lines.append(f"\n### Terrain Classification\n- Detected terrain type: **{terrain}**")

        if metrics:
            lines.append("\n### Model Performance (training validation set)\n")
            for e in comp:
                if e in metrics:
                    lines.append(f"- {e}: MAE={metrics[e].get('mae', 'N/A'):.4f}, R²={metrics[e].get('r2', 'N/A'):.4f}")

        prompt = "\n".join(lines)
        prompt += """

---
## Report Requirements

Generate a professional scientific soil analysis report in Markdown with these sections:

### 1. Executive Summary (2–3 sentences)
Key findings: which elements are elevated, overall contamination level, primary concern.

### 2. Heavy Metal Contamination Assessment
For each element (Cd, Cu, Ni, Mn, Fe, Zn):
- Compare to safe thresholds (WHO/FAO guidelines)
- Classify: Background / Moderate / High / Critical
- Likely source (industrial, natural geogenic, agricultural, etc.)

### 3. Soil Health Classification
Overall contamination index: Clean / Slightly Contaminated / Moderately Contaminated / Heavily Contaminated
Justify with the multi-element pattern.

### 4. Environmental Risk Analysis
- Which elements pose the highest risk to:
  - Human health (direct ingestion/inhalation)
  - Ecosystems (soil biota, plant uptake)
  - Groundwater (leaching potential)

### 5. Probable Source Analysis
Based on the element ratios (e.g. Cd/Zn for smelter, Cu/Ni for mining), infer likely contamination sources.
Reference known Pakistani industrial/agricultural contexts if relevant.

### 6. Remediation Recommendations
3–4 actionable recommendations for soil remediation or management.

### 7. Confidence & Limitations
Note that values are AI-predicted from imagery (not direct chemical assay), with approximate uncertainty ranges.

---
Start directly with: ## SUPARCO SOIL ANALYSIS REPORT
Write in professional scientific language. Be specific with numbers.
"""
        return prompt

    def _fallback_report(self, comp: Dict, terrain: Optional[str]) -> str:
        """Rule-based fallback when Gemini is unavailable."""
        import datetime
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')

        # Contamination scoring
        elevated = []
        critical = []
        for e, v in comp.items():
            thresh = SAFE_THRESHOLDS.get(e, 1e9)
            if v > thresh * 2:
                critical.append(e)
            elif v > thresh:
                elevated.append(e)

        if critical:
            overall = "**Heavily Contaminated** — critical levels detected"
            health_risk = "High — immediate assessment recommended"
        elif elevated:
            overall = "**Moderately Contaminated** — above-threshold levels present"
            health_risk = "Moderate — monitoring and mitigation advised"
        else:
            overall = "**Within acceptable limits** — no critical contamination detected"
            health_risk = "Low — routine monitoring sufficient"

        rows = "\n".join(
            f"| {ELEMENT_FULL[e]} ({e}) | {v:.3f} | {SAFE_THRESHOLDS.get(e,'N/A')} | "
            f"{'⚠️ ELEVATED' if v > SAFE_THRESHOLDS.get(e,1e9) else '✅ Normal'} |"
            for e, v in comp.items()
        )

        remediation = []
        if 'Cd' in critical or 'Cd' in elevated:
            remediation.append("- **Cd**: Phytoremediation using *Thlaspi caerulescens*; restrict food crop cultivation")
        if 'Cu' in critical or 'Ni' in critical:
            remediation.append("- **Cu/Ni**: Soil washing or immobilisation with lime + organic matter")
        if not remediation:
            remediation.append("- Continue regular monitoring; maintain organic matter content")
            remediation.append("- Avoid heavy industrial activities in the vicinity")

        terrain_line = f"\n**Terrain Type**: {terrain}" if terrain else ""

        return f"""## SUPARCO SOIL ANALYSIS REPORT
**Date**: {now}
**Analysis System**: SUPARCO AI Soil Analysis v2.0
**Method**: CNN Regression from Soil Imagery + Rule-Based Assessment{terrain_line}

---

### 1. Executive Summary
AI analysis of the submitted soil sample image yielded heavy metal concentration predictions.
Overall soil health classification: {overall}.
{"Elements of concern: " + ", ".join(critical + elevated) + "." if critical or elevated else "No critical contamination detected."}

---

### 2. Heavy Metal Contamination Assessment

| Element | Predicted (mg/kg) | Safe Limit (mg/kg) | Status |
|---------|------------------|-------------------|--------|
{rows}

---

### 3. Soil Health Classification
{overall}

---

### 4. Environmental Risk Analysis
- **Human Health Risk**: {health_risk}
- **Ecosystem Risk**: {"High — heavy metal bioaccumulation risk in soil biota" if critical else "Moderate — monitor plant uptake pathways"}
- **Groundwater Risk**: {"High — Cd and Zn are mobile under acidic conditions" if 'Cd' in (critical+elevated) or 'Zn' in (critical+elevated) else "Low — elements largely immobile at neutral pH"}

---

### 5. Probable Source Analysis
The detected element pattern is consistent with:
{"- Industrial/smelter contamination (elevated Cd, Zn ratio)" if 'Cd' in (critical+elevated) else "- Natural geogenic background"}
{"- Agricultural runoff (Cu, Mn, Zn from fertilisers/pesticides)" if any(e in (critical+elevated) for e in ['Cu','Mn','Zn']) else ""}
- Further source apportionment requires isotopic analysis (Pb, Sr isotopes)

---

### 6. Remediation Recommendations
{chr(10).join(remediation)}
- Implement soil pH management (target pH 6.5–7.0) to reduce metal mobility
- Conduct spatial sampling to delineate contamination extent

---

### 7. Confidence & Limitations
- Values are AI-predicted from RGB imagery using ResNet-18 regression
- Typical uncertainty: ±0.05–0.5 mg/kg depending on element and concentration range
- Confirm with ICP-MS/OES laboratory analysis before remediation decisions

---
*Report generated by SUPARCO AI Soil Analysis System*
*Model: ResNet-18 | Data: SUPARCO Soil Academia Dataset (180 samples)*
"""
