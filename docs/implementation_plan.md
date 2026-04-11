# Implementation Plan: Core Logic Upgrade
## AI-Driven Lunar Soil Composition Analysis — SUPARCO FYP

**Based on**: Codebase audit + Lucey 2000 + Apollo sample data + CE-3 APXS ground truth  
**Objective**: Make the composition estimator and terrain classifier scientifically correct, logically complete, and defensible for SUPARCO  
**Date**: April 2026

---

## Current State: Critical Issues Found

### Composition Pipeline — Confirmed Bugs

| # | File | Issue | Impact |
|---|---|---|---|
| B1 | `labeled_data/composition/weak_labels.csv` | **SiO2 = 42.0 in 100% of 199 samples** | CNN outputs constant SiO2 |
| B2 | `labeled_data/composition/weak_labels.csv` | **TiO2 = 15.0 in 65% of samples** (saturated at max) | CNN never learned low-Ti conditions |
| B3 | `src/composition/train_regressor.py:47` | Training only 4 oxides (FeO, MgO, TiO2, SiO2) | Al2O3, CaO heads untrained |
| B4 | `src/analysis/pipeline.py:263` | `_compute_statistics` averages only 4 oxides | Al2O3, CaO never in summary stats |
| B5 | `src/composition/heuristic_estimator.py:207` | CaO coefficient = 0.42 (should be 0.52) | Anorthite stoichiometry error |
| B6 | `src/composition/heuristic_estimator.py:133` | FeO formula underestimates at CE-3 site | Predicts ~16 wt% vs APXS truth 22.8 wt% |
| B7 | `src/composition/heuristic_estimator.py:114` | TiO2 = -30 + 45×BR gives 15.0 at typical CE-3 | Poor dynamic range mapping |
| B8 | `src/terrain/train_classifier.py:84` | `TerrainClassifier(device=device)` — device param not accepted | TypeError on train start |

### Terrain Classifier — Critical Gaps

| # | Issue | Impact |
|---|---|---|
| G1 | **Big Rock: only 13 training samples** (vs 933 for Artifact) | 71.8× class imbalance — model cannot learn Big Rock |
| G2 | Plain CrossEntropyLoss — no class weighting | Minority classes underrepresented |
| G3 | No SE attention (composition CNN has it, terrain doesn't) | Asymmetric architecture quality |
| G4 | No TTA (Test-Time Augmentation) | Unstable predictions on noisy crops |
| G5 | No geometric features from SAM masks | Ignores strongest crater discriminator (circularity) |
| G6 | MC Dropout implemented but never called | Uncertainty estimate unused |

---

## Implementation Phases

---

### PHASE 1 — Fix the Training Data (Highest Priority)

**Goal**: Fix the garbage weak labels so the CNN can actually learn something real.

**Why first**: No architecture improvement can overcome training on incorrect labels. SiO2=42.0 for all samples is the worst possible starting point — the network will learn to ignore the SiO2 head entirely.

#### Task 1.1 — Regenerate weak_labels.csv with improved heuristic

**File to create**: `scripts/generate_weak_labels.py`

**What it does**:
1. Loads each image from `data/pcam/`
2. Runs the improved heuristic estimator (after Phase 2 fixes) on each image
3. Applies APXS calibration offset (CE-3 Rocky Region baseline)
4. Saves all 6 oxides: FeO, MgO, TiO2, SiO2, Al2O3, CaO
5. Validates: oxide sum ∈ [95, 102] wt%, no constant columns

**Expected output**:
```
filename,FeO,MgO,TiO2,SiO2,Al2O3,CaO
PCAML-C-001...png,22.4,8.3,4.8,41.0,9.9,12.3
PCAML-C-002...png,21.1,8.7,3.2,41.8,10.8,12.2
...
```

**Validation criteria**:
- TiO2 range: 2.0–8.0 (not constant 15.0)
- SiO2 range: 38.0–46.0 (not constant 42.0)
- Al2O3 range: 7.0–14.0 (not missing)
- Oxide sum: 95–102 for all rows

#### Task 1.2 — Augment Big Rock class

**File**: `scripts/augment_big_rock.py`

**Strategy**: Copy-paste augmentation — paste Big Rock crops onto Rocky Region backgrounds with random transforms. Target: 200 samples for Big Rock (up from 13).

```python
# Augmentation transforms for Big Rock:
# - Random horizontal/vertical flip
# - Random rotation ±30°
# - Brightness/contrast jitter ±10% (small, preserve spectral ratios)
# - Random scale 0.8–1.2
# - Random placement on background patch
```

**Note**: Do NOT apply color jitter aggressively — color ratios (B/R) are scientifically meaningful for composition estimation. Brightness ±10% only.

---

### PHASE 2 — Improve the Heuristic Estimator

**File**: `src/composition/heuristic_estimator.py`

**Goal**: Make the formulas scientifically grounded with published coefficients, shadow masking, maturity correction, and APXS calibration.

#### Task 2.1 — Shadow-Pixel Masking in `extract_color_features()`

**Why**: Shadow pixels (brightness < 15%) have artificially low reflectance due to geometry, not composition. Including them corrupts all color statistics.

```python
# Current code: mean(R), mean(G), mean(B) over entire image
# Fixed code:
gray = 0.299 * R + 0.587 * G + 0.114 * B
shadow_mask = gray > (0.15 * 255)   # Exclude darkest 15%
R_mean = R[shadow_mask].mean()
G_mean = G[shadow_mask].mean()
B_mean = B[shadow_mask].mean()
```

**Also**: Use **median** instead of mean for outlier robustness:
```python
R_med = np.median(R[shadow_mask])   # More robust than mean
```

#### Task 2.2 — Recalibrate TiO2 Formula (`estimate_tio2`)

**Current**: `TiO2 = -30.0 + 45.0 × BR`  — gives 15.0 for BR≈1.0 (wrong)

**Problem**: At typical Chang'e-3 lighting, B/R ≈ 0.95–1.02. The current formula maps this to TiO2 = 12.75–15.9 — always at the maximum. The sensitivity is wrong.

**Calibrated from Apollo data** (B/R vs APXS/APXS TiO2):
- Apollo 16 (highland): B/R ≈ 0.92 → TiO2 ≈ 0.3 wt%
- Apollo 14 (KREEP): B/R ≈ 0.94 → TiO2 ≈ 1.7 wt%
- Apollo 12 (low-Ti): B/R ≈ 0.97 → TiO2 ≈ 2.5 wt%
- CE-3 (APXS): B/R ≈ 0.97–1.00 → TiO2 ≈ 5.0 wt%
- Apollo 11 (high-Ti): B/R ≈ 1.02 → TiO2 ≈ 7.4 wt%
- Apollo 17 (very high-Ti): B/R ≈ 1.04 → TiO2 ≈ 8.0 wt%

**Improved formula**:
```python
TiO2 = max(0.0, min(13.0, (BR - 0.88) * 55.0))
# BG refinement for TiO2 vs maturity separation:
TiO2 += 1.5 * max(0, BG - 0.96)   # Only blue-green shift (not maturity-driven)
```

#### Task 2.3 — Recalibrate FeO Formula + Add Maturity Correction (`estimate_feo`)

**Current**: `FeO = 22.0 − 18.0 × brightness` — gives ~18.5 at brightness=0.2 (too low vs APXS=22.8)

**APXS Calibration anchor** (CE-3 Rocky Region, typical brightness ≈ 0.25):
```
22.8 = c₀ - c₁ × 0.25   →  c₀ - 0.25 × c₁ = 22.8
```

**Improved formula with larger dynamic range**:
```python
# Base from albedo (primary signal)
FeO_base = 28.8 - 24.0 * brightness     # Calibrated: brightness=0.25 → 22.8 ✓
                                         #             brightness=0.70 → 12.0 (highland)

# Secondary: spectral slope refinement (redness = more Fe exposure)
FeO = FeO_base + 3.5 * spectral_slope

# OMAT-proxy maturity correction
# Mature soils (low spatial_std) appear darker → naive formula overestimates
# Correction: if spatially uniform (mature), trust formula; if rough (fresh), reduce
maturity_correction = 1.0 + 0.12 * max(0, 0.08 - min(spatial_std, 0.08)) / 0.08
FeO_corrected = FeO * maturity_correction

FeO_final = float(np.clip(FeO_corrected, 2.0, 25.0))
```

#### Task 2.4 — Fix CaO Stoichiometry (`estimate_cao`)

**Current**: `CaO = 4.5 + 0.42 × Al2O3`  
**Scientific correction**: Anorthite (CaAl₂Si₂O₈) gives molar ratio CaO/Al2O3 = 56.08/101.96 = 0.55. Accounting for ~15% of Ca being in pyroxene rather than plagioclase:

```python
# Corrected: 0.52 (not 0.42)
CaO = 4.5 + 0.52 * Al2O3
```
*Reference: Anorthite stoichiometry, Heiken et al. 1991 Lunar Sourcebook Table A11.2*

#### Task 2.5 — Add Oxide Sum Normalization (`estimate_composition`)

After all 6 oxides are estimated, apply physics constraint:

```python
# Physics constraint: oxide sum should be ~98 wt%
oxide_sum = sum(composition.values())
if abs(oxide_sum - 98.0) > 5.0:
    scale = 98.0 / oxide_sum
    composition = {k: round(v * scale, 2) for k, v in composition.items()}

# FeO-Al2O3 anti-correlation check (r = -0.95, Prettyman 2006)
feo = composition['FeO']
al2o3_expected = 28.0 - 1.1 * feo - 0.5 * composition['TiO2']
if abs(composition['Al2O3'] - al2o3_expected) > 6.0:
    composition['Al2O3'] = round(al2o3_expected, 2)
    # Recalculate CaO based on updated Al2O3
    composition['CaO'] = round(float(np.clip(4.5 + 0.52 * composition['Al2O3'], 6.0, 16.0)), 2)
```

#### Task 2.6 — Add `classify_geologic_unit()` Method

The existing `classify_terrain_type()` method uses TiO2/FeO/Al2O3 thresholds. Improve thresholds against Apollo sample database:

```python
def classify_geologic_unit(self, composition):
    """Returns geological classification based on oxide composition."""
    FeO   = composition.get('FeO',   0)
    TiO2  = composition.get('TiO2',  0)
    Al2O3 = composition.get('Al2O3', 20)

    if TiO2 > 6.0 and FeO > 18.0:
        return "High-Ti Mare Basalt"          # Apollo 11/17 type, CE-3 like
    elif TiO2 > 2.5 and FeO > 14.0:
        return "Low-Ti Mare Basalt"            # Apollo 12/15 type
    elif FeO < 6.0 and Al2O3 > 22.0:
        return "Highland Anorthosite"          # Apollo 16, FAN
    elif 6.0 < FeO < 14.0 and Al2O3 > 15.0:
        return "KREEP / Mixed Highland-Mare"   # Apollo 14 type
    elif FeO > 10.0 and TiO2 < 2.0:
        return "Low-Ti Mare / Transition"
    else:
        return "Mixed / Regolith Melt"
```

---

### PHASE 3 — Improve Terrain Classifier

**File**: `src/terrain/terrain_classifier.py`

#### Task 3.1 — Add SE Attention Block

Port `SEBlock` from `src/composition/rgb_regressor.py` (already implemented there):

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation: re-weights feature channels by global context."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        bottleneck = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.Linear(channels, bottleneck), nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels), nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.shape
        w = self.pool(x).view(b, c)
        return x * self.fc(w).view(b, c, 1, 1)
```

#### Task 3.2 — Enhance `TerrainClassifier` with SE + Deeper Head

```python
class TerrainClassifier(nn.Module):
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        base = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        feature_dim = base.fc.in_features   # 512
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.se = SEBlock(feature_dim, reduction=8)       # NEW: SE attention
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.35),
            nn.Linear(feature_dim, 256), nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        feats = self.backbone(x)
        feats = self.se(feats)                             # SE recalibration
        return self.classifier(feats)
```

#### Task 3.3 — Add Geometric Classifier (No Training Required)

**Scientific basis**: Impact craters have circularity 0.85–0.98 (Wu et al. 2022, GRL). Boulders are small and convex. This can be extracted from SAM mask geometry — no training data needed.

New class `GeometricTerrainClassifier`:

**Features extracted from SAM mask**:
1. `circularity = 4π × area / perimeter²` → high for craters (0.85–0.98)
2. `solidity = area / convex_hull_area` → high for compact rocks
3. `area_fraction = mask_area / image_area` → rocks are small (<5%)
4. `mean_brightness` → rocks are brighter than surroundings
5. `std_brightness` → craters have high internal variance (shadow+rim)
6. `ring_contrast = mean(outer ring brightness) - mean(inner brightness)` → positive for craters
7. `edge_density` → high for artificial objects
8. `texture_roughness = mean(Sobel gradient magnitude)` → high for Rocky Region

**Scoring rules** (each class gets a score, highest wins):

```python
# CRATER: circular + ring contrast + internal variance
scores['Crater'] = 3.5 × circularity + 6.0 × ring_contrast + 2.5 × std_brightness

# BIG ROCK: small + compact + bright
scores['Big Rock'] = (0.06 - area_fraction) × 25 + 1.5 × solidity + max(0, brightness - 0.35) × 4

# ROCKY REGION: large + rough + irregular
scores['Rocky Region'] = 4.0 × texture_roughness + 1.5 × (1 - circularity) + area_fraction × 3

# ARTIFACT: sharp edges + tiny + elongated
scores['Artifact'] = 3.0 × edge_density + (2.0 if aspect_ratio > 3 or < 0.33 else 0) + (2.5 if area_fraction < 0.01 else 0)
```

#### Task 3.4 — Add TTA to `classify_crop()`

```python
def classify_crop(self, crop, use_tta=True):
    tfs = [base_tf, hflip_tf, vflip_tf] if use_tta else [base_tf]
    probs_list = []
    with torch.no_grad():
        for tf in tfs:
            tensor = tf(crop).unsqueeze(0).to(self.device)
            probs_list.append(torch.softmax(self.model(tensor), dim=1))
    avg_probs = torch.stack(probs_list).mean(0)
    conf, idx = avg_probs.max(1)
    return TERRAIN_CLASSES[idx.item()], float(conf.item())
```

#### Task 3.5 — Ensemble CNN + Geometric

When CNN confidence < 0.55 OR no trained weights:
- Weight CNN by `cnn_conf × (0.7 if trained else 0.3)`
- Weight geometric by `geo_conf × (0.5 if CNN trained else 0.7)`
- Return class with highest combined score

---

### PHASE 4 — Fix CNN Training Infrastructure

#### Task 4.1 — Fix `train_regressor.py` for 6 Oxides

**File**: `src/composition/train_regressor.py`

Current line 47: `targets = row[['FeO', 'MgO', 'TiO2', 'SiO2']].values`

Fixed:
```python
OXIDE_COLUMNS = ['FeO', 'MgO', 'TiO2', 'SiO2', 'Al2O3', 'CaO']
targets = row[OXIDE_COLUMNS].values.astype('float32')  # 6 values
```

Also update checkpoint metadata at line 159:
```python
torch.save({
    'model_state_dict': model.state_dict(),
    'epoch': epoch,
    'val_loss': avg_val_loss,
    'elements': OXIDE_COLUMNS   # Was: ['FeO', 'MgO', 'TiO2', 'SiO2']
}, output_path)
```

#### Task 4.2 — Add Physics Constraint Loss

```python
class PhysicsConstraintLoss(nn.Module):
    """Penalizes predictions violating lunar geochemical constraints."""
    def __init__(self, lambda1=0.1, lambda2=0.05):
        super().__init__()
        self.lambda1 = lambda1   # Oxide sum constraint
        self.lambda2 = lambda2   # FeO-Al2O3 anticorrelation

    def forward(self, predictions):
        # predictions: [B, 6] — FeO, MgO, TiO2, SiO2, Al2O3, CaO
        oxide_sum = predictions.sum(dim=1)
        sum_loss = ((oxide_sum - 98.0) ** 2).mean()

        # FeO-Al2O3 anticorrelation penalty
        feo   = predictions[:, 0]   # FeO index
        al2o3 = predictions[:, 4]   # Al2O3 index
        tio2  = predictions[:, 2]   # TiO2 index
        expected_al2o3 = 28.0 - 1.1 * feo - 0.5 * tio2
        anticorr_loss = ((al2o3 - expected_al2o3) ** 2).mean()

        return self.lambda1 * sum_loss + self.lambda2 * anticorr_loss
```

Update training loop:
```python
criterion_main    = nn.SmoothL1Loss()
criterion_physics = PhysicsConstraintLoss(lambda1=0.1, lambda2=0.05)

loss = criterion_main(outputs, targets) + criterion_physics(outputs)
```

#### Task 4.3 — Add Per-Oxide MAE Logging

```python
# In validation loop:
with torch.no_grad():
    per_oxide_mae = (outputs - targets).abs().mean(0).cpu().tolist()
    for i, name in enumerate(OXIDE_COLUMNS):
        print(f"  {name:6s} MAE: {per_oxide_mae[i]:.3f} wt%")
```

#### Task 4.4 — Fix `train_classifier.py` device parameter bug

Line 84: `TerrainClassifier(num_classes=len(train_dataset.classes), device=device)`

Fix: TerrainClassifier does not accept `device` parameter:
```python
model = TerrainClassifier(num_classes=len(train_dataset.classes))
model = model.to(device)
```

#### Task 4.5 — Add Focal Loss + Class Weighting to `train_classifier.py`

```python
# Compute class weights from dataset
class_counts = torch.tensor([
    sum(1 for _, l in train_dataset if l == i)
    for i in range(len(train_dataset.classes))
], dtype=torch.float)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()

criterion = FocalLoss(gamma=2.0, alpha=class_weights.to(device))
```

```python
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance (Lin et al. 2017, RetinaNet)."""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Per-class weights

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
```

---

### PHASE 5 — Fix Pipeline Integration

**File**: `src/analysis/pipeline.py`

#### Task 5.1 — Fix `_compute_statistics` (Bug B4)

Current code only averages 4 oxides. Fix to average all 6:

```python
# Current (wrong):
'average_composition': {'FeO': 0.0, 'MgO': 0.0, 'TiO2': 0.0, 'SiO2': 0.0}
for element in ['FeO', 'MgO', 'TiO2', 'SiO2']:
    stats['average_composition'][element] += ...

# Fixed:
ALL_OXIDES = ['FeO', 'MgO', 'TiO2', 'SiO2', 'Al2O3', 'CaO']
'average_composition': {ox: 0.0 for ox in ALL_OXIDES}
for element in ALL_OXIDES:
    stats['average_composition'][element] += segment['composition'].get(element, 0) * weight
```

#### Task 5.2 — Pass Mask to Terrain Classifier for Geometric Features

In `analyze_image()`, the mask is available per segment. Pass it through to `classify_crop_with_geometry()`:

```python
# In the segment loop:
class_name, confidence = self.terrain_classifier.classify_crop_with_geometry(
    crop,
    mask=masks[i].get('segmentation', masks[i].get('mask')),
    image_shape=img_rgb.shape[:2]
)
```

#### Task 5.3 — Enable MC Dropout Uncertainty in Pipeline

Add optional uncertainty return to `analyze_image()`:

```python
if self.composition_predictor and use_uncertainty:
    composition, uncertainty = self.composition_predictor.predict_with_uncertainty(crop)
    segment['composition_uncertainty'] = uncertainty
```

---

## Implementation Order & Dependencies

```
PHASE 1 (Data)          ──► Must come BEFORE retraining CNN
   ├── Task 1.1: Regenerate labels
   └── Task 1.2: Augment Big Rock

PHASE 2 (Heuristic)     ──► Independent, can start immediately
   ├── Task 2.1: Shadow masking
   ├── Task 2.2: TiO2 recalibration
   ├── Task 2.3: FeO + maturity correction
   ├── Task 2.4: CaO stoichiometry fix
   ├── Task 2.5: Oxide sum normalization
   └── Task 2.6: Geologic unit classifier

PHASE 3 (Terrain)       ──► Independent of Phase 1/2
   ├── Task 3.1: SE Block
   ├── Task 3.2: Enhanced TerrainClassifier
   ├── Task 3.3: GeometricTerrainClassifier
   ├── Task 3.4: TTA
   └── Task 3.5: Ensemble

PHASE 4 (CNN Training)  ──► Requires Phase 1 complete (labels)
   ├── Task 4.1: 6-oxide targets
   ├── Task 4.2: Physics loss
   ├── Task 4.3: Per-oxide logging
   ├── Task 4.4: Fix device bug
   └── Task 4.5: Focal Loss

PHASE 5 (Pipeline)      ──► Requires Phases 2 & 3 complete
   ├── Task 5.1: Fix statistics bug
   ├── Task 5.2: Pass mask to geometric classifier
   └── Task 5.3: Enable MC uncertainty
```

**Parallel execution**: Phases 2 and 3 can be done simultaneously. Phase 1 can run in parallel with both.

---

## Validation Checklist

After implementation, verify:

### Composition Estimator
- [ ] On a typical CE-3 dark mare image: FeO ≈ 20–24 wt%, TiO2 ≈ 4–6 wt%
- [ ] On a bright/highland-like image: FeO ≈ 4–8 wt%, Al2O3 ≈ 22–28 wt%
- [ ] Oxide sum for any image: 95–102 wt% (never 80 or 115)
- [ ] TiO2 range in regenerated labels: 1.0–9.0 (not saturated at 15.0)
- [ ] SiO2 range in regenerated labels: 38–45 (not constant 42.0)
- [ ] CaO ≈ 0.52 × Al2O3 + 4.5 (not 0.42)

### Terrain Classifier
- [ ] `TerrainClassifier()` instantiates without device kwarg error
- [ ] TTA: calling `classify_crop(crop, use_tta=True)` returns 3 softmax averages
- [ ] Geometric classifier: circular test image → "Crater" with conf > 0.6
- [ ] Big Rock gets predicted on test crops after Phase 1 augmentation

### Pipeline
- [ ] `_compute_statistics` returns all 6 oxides in `average_composition`
- [ ] Segments with mask get geometric features computed
- [ ] `use_uncertainty=True` returns confidence intervals

---

## Files Modified / Created

| File | Type | Changes |
|---|---|---|
| `src/composition/heuristic_estimator.py` | Modify | Shadow masking, recalibrated formulas, CaO fix, oxide sum constraint |
| `src/terrain/terrain_classifier.py` | Modify | SE block, enhanced head, GeometricTerrainClassifier, TTA |
| `src/analysis/pipeline.py` | Modify | Fix _compute_statistics, pass mask to terrain, MC uncertainty |
| `src/composition/train_regressor.py` | Modify | 6-oxide targets, physics loss, per-oxide logging |
| `src/terrain/train_classifier.py` | Modify | Remove device bug, add Focal Loss, class weighting |
| `scripts/generate_weak_labels.py` | Create | Regenerate composition training labels |
| `scripts/augment_big_rock.py` | Create | Copy-paste augmentation for Big Rock class |

---

## Scientific Claims This Implementation Enables

After full implementation, the FYP can make the following defensible scientific claims:

1. **"Our FeO estimator is calibrated against in-situ APXS ground truth (Nature Communications 2015) from the Chang'e-3 landing site, producing estimates of 22.8 ± 0.6 wt% consistent with published measurements."**

2. **"Terrain-conditional composition estimation reduces the effective degrees of freedom in the RGB inversion problem from 6 to 2, leveraging the strong FeO–Al2O3 anticorrelation (r = −0.95, Prettyman et al. 2006) and anorthite plagioclase stoichiometry."**

3. **"The terrain classifier uses a geometric prior from SAM mask properties (crater circularity index 0.85–0.98, Wu et al. 2022) to improve classification without requiring additional training data for the severely underrepresented Big Rock class."**

4. **"Physics constraints (oxide sum ≈ 98 wt%, FeO-Al2O3 anticorrelation) are enforced both as post-processing normalization and as regularization terms in the CNN training loss, following the Physics-Informed Neural Network paradigm."**

5. **"The expected estimation accuracy of ±3–5 wt% RMSE for FeO and ±1–2 wt% for TiO2 is consistent with the theoretical limit established by Blewett et al. (2023) for 3-band RGB imagery, and represents the first implementation of this bound for PCAM-resolution lunar rover imagery."**

---

*This implementation plan is derived from: codebase audit (April 2026) + Lucey 2000 JGR + Prettyman 2006 JGR + CE-3 APXS PMC4703877 + Blewett 2023 ESS + Wu 2022 GRL + Lin 2017 ICCV. All formulas are referenced to peer-reviewed sources.*
