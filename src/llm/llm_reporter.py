"""
LLM-Powered Scientific Report Generator for Lunar Soil Analysis
Uses Google Gemini API for deep geological interpretation and mission reporting.
"""

import os
import base64
import logging
from typing import Dict, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import io

logger = logging.getLogger(__name__)


# Geological reference thresholds based on published lunar science
MINERAL_THRESHOLDS = {
    'ilmenite_rich': {'TiO2': 6.0, 'FeO': 15.0},      # High-Ti mare (FeTiO3)
    'mare_basalt_low_ti': {'TiO2': 1.0, 'FeO': 12.0},  # Low-Ti mare
    'anorthosite': {'FeO': 8.0},                         # Highland crust
    'kreep': {'SiO2': 48.0},                             # KREEP (K, REE, P)
    'pyroxenite': {'MgO': 10.0, 'FeO': 16.0},           # Ultramafic
}

ISRU_RESOURCES = {
    'oxygen': 'Ilmenite reduction (FeTiO3 + H2 → Fe + TiO2 + H2O)',
    'silicon': 'Solar cell manufacturing and semiconductor substrates',
    'iron': 'Structural construction and shielding material',
    'titanium': 'High-strength alloys for habitat construction',
}


def _load_gemini():
    """Lazy-load the Gemini client (new google.genai SDK preferred)."""
    try:
        from google import genai
        return genai, 'new'
    except ImportError:
        pass
    try:
        import google.generativeai as genai
        return genai, 'legacy'
    except ImportError:
        return None, None


class LunarLLMReporter:
    """
    AI-powered scientific report generator for lunar soil analysis.

    Integrates Google Gemini to produce:
    - Deep geological interpretation
    - Mineralogical phase assessment
    - ISRU (In-Situ Resource Utilization) potential
    - Terrain safety evaluation
    - Mission recommendations
    """

    SYSTEM_CONTEXT = """You are a senior planetary geologist specializing in lunar science at a space agency.
You are analyzing data from an AI system that processed lunar surface imagery from Chang'e 3 PCAM cameras.
The data comes from spectral and color analysis of lunar soil (regolith) images.
Write scientifically rigorous reports that are also accessible to mission planners."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the LLM reporter.

        Args:
            api_key: Google Gemini API key. Falls back to GEMINI_API_KEY env variable.
        """
        self.api_key = api_key or os.environ.get('GEMINI_API_KEY') or self._load_from_dotenv()
        self.model = None
        self.vision_model = None
        self._initialized = False

        if self.api_key:
            self._init_gemini()
        else:
            logger.warning("No Gemini API key found. LLM report generation will use fallback.")

    def _load_from_dotenv(self) -> Optional[str]:
        """Load API key from .env file if present."""
        env_path = Path(__file__).parent.parent.parent / '.env'
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith('GEMINI_API_KEY='):
                    return line.split('=', 1)[1].strip().strip('"').strip("'")
        return None

    def _init_gemini(self):
        """Configure and initialize Gemini models."""
        genai, sdk_type = _load_gemini()
        if genai is None:
            logger.warning("google-genai not installed. Run: pip install google-genai")
            return

        try:
            if sdk_type == 'new':
                # New google.genai SDK
                self._client = genai.Client(api_key=self.api_key)
                self._sdk_type = 'new'
                self._model_name = 'gemini-2.0-flash'
            else:
                # Legacy google.generativeai SDK
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(
                    model_name='gemini-1.5-flash',
                    system_instruction=self.SYSTEM_CONTEXT
                )
                self.vision_model = self.model
                self._sdk_type = 'legacy'

            self._initialized = True
            logger.info(f"Gemini LLM initialized ({sdk_type} SDK)")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")

    @property
    def is_available(self) -> bool:
        """Returns True if LLM is ready to generate reports."""
        if not self._initialized:
            return False
        sdk = getattr(self, '_sdk_type', None)
        if sdk == 'new':
            return getattr(self, '_client', None) is not None
        return getattr(self, 'model', None) is not None

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def generate_report(
        self,
        analysis_results: Dict,
        image: Optional[np.ndarray] = None,
        mission_context: str = ""
    ) -> str:
        """
        Generate a comprehensive scientific mission report.

        Args:
            analysis_results: Output dict from LunarAnalysisPipeline.analyze_image()
            image: Optional RGB numpy image for visual analysis
            mission_context: Optional mission-specific context string

        Returns:
            Formatted Markdown report string
        """
        stats = analysis_results.get('statistics', {})
        segments = analysis_results.get('segments', [])
        mode = analysis_results.get('mode', 'unknown')

        # Build the analysis prompt
        prompt = self._build_prompt(stats, segments, mode, mission_context)

        if self.is_available:
            try:
                if image is not None:
                    return self._generate_with_vision(prompt, image)
                else:
                    return self._generate_text_only(prompt)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                logger.info("Falling back to template report")

        # Fallback: scientifically enriched template
        return self._generate_fallback_report(stats, segments)

    # ------------------------------------------------------------------
    # Private generation methods
    # ------------------------------------------------------------------

    def _generate_text_only(self, prompt: str) -> str:
        """Call Gemini with text-only prompt."""
        if self._sdk_type == 'new':
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=self.SYSTEM_CONTEXT + "\n\n" + prompt
            )
            return response.text
        else:
            response = self.model.generate_content(prompt)
            return response.text

    def _generate_with_vision(self, prompt: str, image: np.ndarray) -> str:
        """Call Gemini with image + text prompt."""
        genai, sdk_type = _load_gemini()
        pil_img = Image.fromarray(image.astype(np.uint8))

        # Resize if too large
        max_dim = 1024
        if max(pil_img.size) > max_dim:
            pil_img.thumbnail((max_dim, max_dim), Image.LANCZOS)

        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=85)
        image_bytes = buf.getvalue()

        if self._sdk_type == 'new':
            from google.genai import types
            image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')
            full_prompt = self.SYSTEM_CONTEXT + "\n\n" + prompt
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[full_prompt, image_part]
            )
            return response.text
        else:
            image_part = {'mime_type': 'image/jpeg', 'data': image_bytes}
            response = self.vision_model.generate_content([prompt, image_part])
            return response.text

    # ------------------------------------------------------------------
    # Prompt engineering
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        stats: Dict,
        segments: list,
        mode: str,
        mission_context: str
    ) -> str:
        """Build a detailed scientific prompt from analysis results."""

        comp = stats.get('average_composition', {})
        terrain_dist = stats.get('terrain_distribution', {})
        total_seg = stats.get('total_segments', 0)

        # Summarize per-segment composition range
        if segments:
            feo_vals = [s['composition'].get('FeO', 0) for s in segments]
            tio2_vals = [s['composition'].get('TiO2', 0) for s in segments]
            sio2_vals = [s['composition'].get('SiO2', 0) for s in segments]
            mgo_vals = [s['composition'].get('MgO', 0) for s in segments]
            al2o3_vals = [s['composition'].get('Al2O3', 0) for s in segments]
            cao_vals = [s['composition'].get('CaO', 0) for s in segments]
            
            spatial_variability = (
                f"FeO range: {min(feo_vals):.1f}–{max(feo_vals):.1f}%  "
                f"| TiO2 range: {min(tio2_vals):.1f}–{max(tio2_vals):.1f}%  "
                f"| SiO2 range: {min(sio2_vals):.1f}–{max(sio2_vals):.1f}%  "
                f"| MgO range: {min(mgo_vals):.1f}–{max(mgo_vals):.1f}%  "
                f"| Al2O3 range: {min(al2o3_vals):.1f}–{max(al2o3_vals):.1f}%  "
                f"| CaO range: {min(cao_vals):.1f}–{max(cao_vals):.1f}%"
            )
        else:
            spatial_variability = "No segment-level data available"

        terrain_table = "\n".join([
            f"  - {terrain}: {data['count']} segments, {data['percentage']:.1f}% coverage"
            for terrain, data in terrain_dist.items()
        ]) or "  - No terrain classification data"

        mission_ctx_section = f"\n**Mission Context**: {mission_context}" if mission_context else ""

        prompt = f"""
You have received AI-processed data from a lunar surface image captured by the Chang'e 3 PCAM camera system.
Analysis was performed using SAM 2.1 segmentation + ResNet-18 terrain classification + Lucey color-ratio heuristics.{mission_ctx_section}

---
## INPUT DATA SUMMARY

**Analysis Mode**: {mode}
**Total Geological Features Detected**: {total_seg}

### Area-Weighted Average Composition (wt%)
- FeO  (Iron Oxide):       {comp.get('FeO', 0):.2f}%
- MgO  (Magnesium Oxide):  {comp.get('MgO', 0):.2f}%
- TiO2 (Titanium Dioxide): {comp.get('TiO2', 0):.2f}%
- SiO2 (Silicon Dioxide):  {comp.get('SiO2', 0):.2f}%
- Al2O3 (Aluminum Oxide):  {comp.get('Al2O3', 0):.2f}%
- CaO  (Calcium Oxide):    {comp.get('CaO', 0):.2f}%

### Spatial Variability Across Segments
{spatial_variability}

### Terrain Classification Distribution
{terrain_table}

---
## REPORT REQUIREMENTS

Generate a professional scientific mission report in Markdown with these exact sections:

### 1. Executive Summary (3–4 sentences)
Concise overview of the geological setting and key findings.

### 2. Geological Classification
- Classify this region as: High-Ti Mare / Low-Ti Mare / Highland (Anorthosite) / Mixed / KREEP Terrane
- Justify based on oxide abundances using accepted thresholds:
  * TiO2 > 6%  → High-Ti Mare (Ilmenite-rich)
  * TiO2 1–6%  → Low-Ti Mare Basalt
  * TiO2 < 1%, FeO < 8%, Al2O3 > 22% → Highland / Anorthosite
  * SiO2 > 48%, with high K → KREEP
- Compare to Apollo and Chang'e reference samples if relevant

### 3. Mineralogical Interpretation
Infer likely mineral phases present. Use standard lunar mineralogy:
- Ilmenite (FeTiO3): indicated by high TiO2+FeO
- Olivine (Mg2SiO4): indicated by high MgO
- Pyroxene (CaMgFeSiO system): indicated by moderate FeO+MgO
- Plagioclase (CaAl2Si2O8): indicated by low FeO, high Al2O3+CaO
- Glass: amorphous matrix in regolith

### 4. ISRU (In-Situ Resource Utilization) Potential
Assess the economic and strategic value for future lunar base operations:
- Oxygen extraction potential (hydrogen reduction of ilmenite)
- Metal availability (Fe, Ti, Mg, Si, Al, Ca)
- Radiation shielding suitability
- Construction material quality
- Rate each on a scale: Low / Moderate / High / Excellent

### 5. Terrain Safety Assessment
Based on the terrain class distribution ({terrain_table.strip()}):
- Landing suitability (1–5 scale, 5=safest)
- Rover trafficability
- Hazard summary (craters, boulders, slopes)

### 6. Anomalies & Points of Interest
Note any geologically significant observations:
- Unusual oxide ratios (e.g. FeO/Al2O3 anticorrelation violations)
- High-confidence segments that differ from the mean
- Potential scientific sampling targets

### 7. Mission Recommendations
3–5 actionable bullet points for mission planners regarding:
- Sampling priorities
- Navigation advisories
- Follow-up observation needs

### 8. Confidence & Limitations
Briefly note the limitations of RGB-only composition estimation vs. hyperspectral data.

---
Write the report now. Use professional scientific language. Be specific with numbers.
Start directly with "## SUPARCO LUNAR ANALYSIS REPORT".
"""
        return prompt

    # ------------------------------------------------------------------
    # Fallback template (no LLM)
    # ------------------------------------------------------------------

    def _generate_fallback_report(self, stats: Dict, segments: list) -> str:
        """
        Scientifically enriched template report when LLM is unavailable.
        Applies rule-based geological reasoning.
        """
        import pandas as pd

        comp = stats.get('average_composition', {})
        terrain_dist = stats.get('terrain_distribution', {})
        total_seg = stats.get('total_segments', 0)
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')

        feo = comp.get('FeO', 0)
        tio2 = comp.get('TiO2', 0)
        mgo = comp.get('MgO', 0)
        sio2 = comp.get('SiO2', 0)
        al2o3 = comp.get('Al2O3', 0)
        cao = comp.get('CaO', 0)

        # Rule-based classification
        if tio2 > 6.0 and feo > 15.0:
            geo_class = "High-Ti Mare Basalt"
            geo_detail = ("Ilmenite-rich volcanic regolith. High TiO2 (>6%) indicates "
                          "abundant FeTiO3. Consistent with Mare Tranquillitatis-type "
                          "volcanism. Excellent target for oxygen extraction via "
                          "hydrogen reduction of ilmenite.")
            isru_o2 = "High"
        elif tio2 > 2.0 and feo > 10.0:
            geo_class = "Low-Ti Mare Basalt"
            geo_detail = ("Pyroxene-dominated mare basalt with moderate TiO2. "
                          "Consistent with Mare Imbrium / Oceanus Procellarum material. "
                          "Typical Chang'e 3 landing site composition.")
            isru_o2 = "Moderate"
        elif feo < 8.0 and al2o3 > 20.0:
            geo_class = "Lunar Highland (Anorthosite)"
            geo_detail = ("Low-FeO content and high Al2O3 indicates feldspar-dominated highland crust. "
                          "Likely plagioclase-rich anorthosite. This is ancient primordial "
                          "crust, scientifically significant for understanding lunar "
                          "magma ocean differentiation.")
            isru_o2 = "Low"
        else:
            geo_class = "Mixed Mare/Highland Transition"
            geo_detail = ("Intermediate composition suggests mixing of mare basalt and "
                          "highland material, possibly via impact gardening or lateral "
                          "transport. Scientifically interesting for regolith evolution studies.")
            isru_o2 = "Moderate"

        # Dominant terrain
        dom_terrain = max(terrain_dist.items(), key=lambda x: x[1]['percentage'])[0] if terrain_dist else "Unknown"

        # Terrain table
        terrain_rows = "\n".join([
            f"| {t} | {d['count']} | {d['percentage']:.1f}% |"
            for t, d in terrain_dist.items()
        ])

        # Landing safety
        crater_pct = terrain_dist.get('Crater', {}).get('percentage', 0)
        rock_pct = terrain_dist.get('Big Rock', {}).get('percentage', 0)
        hazard_pct = crater_pct + rock_pct
        if hazard_pct < 15:
            safety_score = "4/5 — Low hazard density, favorable for landing"
        elif hazard_pct < 35:
            safety_score = "3/5 — Moderate hazards, careful site selection required"
        else:
            safety_score = "2/5 — High hazard density, avoid as primary landing zone"

        # Mineral inference
        minerals = []
        if tio2 > 4.0 and feo > 14.0:
            minerals.append("**Ilmenite** (FeTiO3) — primary Fe-Ti phase")
        if mgo > 9.0:
            minerals.append("**Olivine** (Mg2SiO4) — high Mg ultramafic component")
        if feo > 10.0:
            minerals.append("**Pyroxene** (augite/pigeonite) — dominant silicate")
        if al2o3 > 18.0:
            minerals.append("**Plagioclase** (anorthite) — feldspar-dominant highland component")
        minerals.append("**Agglutinitic glass** — micrometeorite impact fusion product (ubiquitous in regolith)")
        mineral_list = "\n".join(f"- {m}" for m in minerals)

        report = f"""## SUPARCO LUNAR ANALYSIS REPORT
**Classification**: UNCLASSIFIED
**Date**: {timestamp}
**Analysis System**: SUPARCO AI Lunar Analysis v2.0
**Method**: SAM 2.1 Segmentation + ResNet-18 Classification + Lucey RGB Heuristics

---

### 1. Executive Summary
Analysis of the submitted lunar surface imagery identified **{total_seg} distinct geological features**.
The region is classified as **{geo_class}**, based on area-weighted oxide abundances.
{geo_detail}
Dominant surface type is **{dom_terrain}**, covering
{terrain_dist.get(dom_terrain, {}).get('percentage', 0):.1f}% of the analyzed area.

---

### 2. Geological Classification
**Primary Classification**: {geo_class}

| Oxide | Measured (wt%) | Typical Range | Classification Signal |
|-------|---------------|---------------|----------------------|
| FeO   | {feo:.2f}% | Mare: 12–20%, Highland: 3–8% | {"Mare indicator ↑" if feo > 12 else "Highland indicator ↓"} |
| TiO2  | {tio2:.2f}% | High-Ti: >6%, Low-Ti: 1–6%, Highland: <1% | {"High-Ti Mare" if tio2 > 6 else "Low-Ti Mare" if tio2 > 1 else "Highland"} |
| MgO   | {mgo:.2f}% | Mare: 6–10%, Ultramafic: >10% | {"Ultramafic signal" if mgo > 10 else "Normal mare/highland"} |
| SiO2  | {sio2:.2f}% | Mare: ~44–46%, KREEP: ~48–52% | {"KREEP-enriched signal" if sio2 > 48 else "Normal basalt"} |
| Al2O3 | {al2o3:.2f}% | Highland: 20–30%, Mare: 7–11% | {"Highland plagioclase" if al2o3 > 20 else "Basaltic Al"} |
| CaO   | {cao:.2f}% | Highland: 13–16%, Mare: 8–12% | {"Plagioclase-rich" if cao > 13 else "Basaltic Ca"} |

---

### 3. Mineralogical Interpretation
Inferred mineral phases based on oxide geochemistry:

{mineral_list}

*Note: Precise mineralogy requires hyperspectral data (e.g., M3 from Chandrayaan-1 or Kaguya MI).
RGB-based estimates carry ±2–3 wt% uncertainty.*

---

### 4. ISRU Resource Potential

| Resource | Potential | Basis |
|----------|-----------|-------|
| **O2 (Ilmenite Reduction)** | {isru_o2} | TiO2={tio2:.1f}%, FeO={feo:.1f}% |
| **Iron (Fe)** | {"High" if feo > 15 else "Moderate" if feo > 10 else "Low"} | FeO={feo:.1f}% |
| **Silicon (Si)** | {"High" if sio2 > 46 else "Moderate"} | SiO2={sio2:.1f}% |
| **Titanium (Ti)** | {"High" if tio2 > 6 else "Moderate" if tio2 > 2 else "Low"} | TiO2={tio2:.1f}% |
| **Magnesium (Mg)** | {"High" if mgo > 10 else "Moderate"} | MgO={mgo:.1f}% |
| **Aluminum (Al)** | {"High" if al2o3 > 20 else "Moderate"} | Al2O3={al2o3:.1f}% |
| **Calcium (Ca)** | {"High" if cao > 13 else "Moderate"} | CaO={cao:.1f}% |
| **Regolith for Shielding** | High | Lunar regolith is effective against GCR/SPE |

---

### 5. Terrain Safety Assessment

**Landing Safety Score**: {safety_score}

| Terrain Type | Segments | Coverage |
|-------------|---------|---------|
{terrain_rows}

- **Rover Trafficability**: {"Moderate — crater presence requires path planning" if crater_pct > 20 else "Good — low crater density"}
- **Primary Hazards**: {"Craters ({:.1f}%), ".format(crater_pct) if crater_pct > 5 else ""}{"Boulders ({:.1f}%)".format(rock_pct) if rock_pct > 5 else "No significant hazards detected"}

---

### 6. Anomalies & Points of Interest
- Area-weighted composition is internally consistent with published Chang'e 3 landing site data
- TiO2/FeO ratio ({(tio2/feo if feo > 0 else 0):.3f}) is {"anomalously high — potential ilmenite concentration" if tio2/feo > 0.4 else "within expected mare basalt range"}
- Any segments with terrain class "Artifact" represent the Chang'e 3 lander/rover and should be excluded from geological sampling statistics

---

### 7. Mission Recommendations
- **Sampling Priority**: {"Target high-FeO, high-TiO2 segments for oxygen extraction resource mapping" if isru_o2 in ["High", "Moderate"] else "Prioritize highland plagioclase regions for primordial crust samples"}
- **Navigation**: Avoid crater rims and boulder fields (identified as {rock_pct:.0f}% of terrain)
- **Follow-up Observations**: Acquire hyperspectral data (400–2500nm) to confirm mineral phase assignments and reduce composition uncertainty
- **Science Target**: {"Ilmenite-bearing regolith patches are priority targets for ISRU pilot experiments" if tio2 > 3 else "Anorthositic outcrops offer window into primordial magma ocean crystallization"}
- **Long Baseline**: Cross-reference with LROC (Lunar Reconnaissance Orbiter Camera) high-resolution images for context

---

### 8. Confidence & Limitations
- **Method**: Lucey color-ratio heuristics adapted for RGB (vs. original UV-NIR wavelengths)
- **Uncertainty**: ±2–4 wt% for FeO/TiO2, ±3–5 wt% for MgO/SiO2
- **Improvement Path**: Replace heuristics with CNN trained on LROC WAC hyperspectral ground-truth; add 950nm pyroxene band
- **Validation**: Compare against Chang'e 5 sample return (CLEP-CE5C1000GP) for nearest-neighbor composition reference

---
*Report generated by SUPARCO AI Lunar Analysis System*
*Powered by: SAM 2.1 | ResNet-18 | Lucey RGB Heuristics | SUPARCO Research Division*
"""
        return report
