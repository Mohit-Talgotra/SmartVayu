# Plus Dataset Analysis - Statistical Tests & Results Explained

## 📊 Generated Visualization Files

### 1. **01_temperature_distribution.png**
**What it shows:**
- Left panel: Temperature histogram with normal curve overlay
- Right panel: Q-Q plot comparing actual data to normal distribution

**Key findings:**
- Distribution is approximately symmetric (slight left skew: -0.235)
- Heavy-tailed (leptokurtic) with kurtosis = 1.333
- Q-Q plot shows deviations from normality, especially in the tails

---

### 2. **02_temperature_by_device.png**
**What it shows:**
- Four different views of temperature variation across RPI devices (20, 21, 30, 39, 50)
- Box plots, violin plots, mean comparison with error bars, overlaid histograms

**Key findings:**
- **Significant device differences:** RPI 50 has highest mean (30.21°C), RPI 30 has lowest (26.50°C)
- Different variability: RPI 20 most consistent (SD=1.95°C), RPI 50 most variable (SD=2.74°C)
- All devices show good data coverage with no major gaps

---

### 3. **03_time_series_analysis.png**
**What it shows:**
- Full time series (sampled), daily averages, monthly statistics, trend analysis
- Temperature patterns over the 384-day study period

**Key findings:**
- Clear seasonal patterns visible
- Some missing data periods (as identified earlier)
- Monthly temperature ranges from ~26°C to ~28°C on average
- Weekly trend shows good temporal coverage

---

### 4. **04_correlation_analysis.png**
**What it shows:**
- Correlation heatmap of all sensors
- Temperature correlations ranked by strength
- Scatter plots of strongest positive and negative correlations

**Key findings:**
- **Strongest negative correlation:** NH3 sensor (r = -0.627)
- **Strongest positive correlation:** sound_low (r = 0.232)
- Gas sensors show strong environmental relationships with temperature

---

### 5. **05_sensor_relationships.png**
**What it shows:**
- Temperature relationships with gas sensors, light, sound, and environmental conditions
- Unique sensors not available in REG dataset (oxidised, reduced, nh3)

**Key findings:**
- Gas sensors show clear inverse relationship with temperature
- Light level shows weak correlation with temperature
- Sound levels have minimal correlation with temperature
- Humidity-pressure plot colored by temperature shows environmental clustering

---

### 6. **06_statistical_diagnostics.png**
**What it shows:**
- Residual analysis, residual distribution, monthly patterns, outlier detection
- Statistical diagnostic plots for model assumptions

**Key findings:**
- Residuals show some temporal patterns (not perfectly random)
- Few statistical outliers detected (Z-score > 3)
- Monthly temperature patterns are consistent
- Some heteroscedasticity (non-constant variance) over time

---

## 🧪 Statistical Tests Performed & Explanations

### **1. Descriptive Statistics**
**Purpose:** Summarize the central tendencies and variability of temperature data

**Results:**
- **Mean:** 27.80°C (average temperature across all measurements)
- **Median:** 27.85°C (middle value when sorted - close to mean indicates symmetric distribution)
- **Standard Deviation:** 2.73°C (typical deviation from the mean)
- **Coefficient of Variation:** 9.84% (low variability = good measurement consistency)
- **Skewness:** -0.235 (slightly left-skewed, but approximately symmetric)
- **Kurtosis:** 1.333 (heavy-tailed distribution, more extreme values than normal)

**Interpretation:**
✅ **Excellent data quality** - Low variability indicates consistent measurements
✅ **Approximately normal shape** - Skewness near 0 suggests symmetric distribution
⚠️ **Heavy tails** - More extreme values than expected in normal distribution

---

### **2. Normality Tests**
**Purpose:** Determine if temperature data follows a normal (bell-curve) distribution

**Tests Performed:**
1. **Shapiro-Wilk Test** (W = 0.988, p < 0.001)
2. **D'Agostino-Pearson Test** (statistic = 182.88, p < 0.001)  
3. **Jarque-Bera Test** (statistic = 439.80, p < 0.001)
4. **Kolmogorov-Smirnov Test** (D = 0.031, p < 0.001)
5. **Anderson-Darling Test** (A = 7.44, critical values exceeded)

**Results:** 0/4 tests suggest normal distribution (all p-values < 0.05)

**Why this matters:**
- **Non-normal data** → Use non-parametric statistical tests
- **Large sample size** → Central Limit Theorem still applies for means
- **Heavy tails** → More extreme values than normal distribution predicts

**Recommendation:** Use non-parametric methods (Spearman correlation, Mann-Whitney U, Kruskal-Wallis)

---

### **3. Correlation Analysis**
**Purpose:** Measure relationships between temperature and other sensor variables

**Pearson Correlations (Linear relationships):**
- **NH3:** r = -0.627 (strong negative correlation)
- **Oxidised gas:** r = -0.282 (moderate negative correlation)
- **Sound_low:** r = 0.232 (weak positive correlation)
- **Sound_amp:** r = 0.139 (weak positive correlation)

**Spearman Correlations (Rank-based relationships):**
- **NH3:** ρ = -0.704 (strong negative correlation)
- **Oxidised gas:** ρ = -0.603 (strong negative correlation)
- **Sound_low:** ρ = 0.364 (moderate positive correlation)

**Key Insights:**
1. **Gas sensors strongly correlate with temperature** → Environmental relationship
2. **NH3 is the strongest predictor** → Higher temperatures = lower NH3 levels
3. **Sound relationships are weak** → Minimal environmental coupling
4. **Spearman > Pearson correlations** → Non-linear relationships exist

---

### **4. Variance Homogeneity Tests**
**Purpose:** Test if temperature variability is equal across different RPI devices

**Tests Performed:**
1. **Levene's Test** (F = 4878.11, p < 0.001)
2. **Bartlett's Test** (statistic = 60394.31, p < 0.001)
3. **Fligner-Killeen Test** (statistic = 21182.14, p < 0.001)

**Results:** All tests significant (p < 0.001) → **Variances are NOT equal**

**What this means:**
- Different RPI devices have different measurement precision
- RPI 20: Most consistent (SD = 1.95°C)
- RPI 50: Most variable (SD = 2.74°C)
- **Impact:** Use Welch's ANOVA or non-parametric tests for group comparisons

---

### **5. Group Comparison Tests**
**Purpose:** Test if mean temperatures differ significantly between RPI devices

**Tests Performed:**
1. **One-Way ANOVA** (F = 125,343.90, p < 0.001)
2. **Kruskal-Wallis Test** (H = 429,263.73, p < 0.001)

**Results:** Both tests highly significant (p < 0.001)

**Device Temperature Means:**
- **RPI 50:** 30.21°C (highest)
- **RPI 39:** 28.82°C
- **RPI 20:** 28.34°C  
- **RPI 21:** 26.62°C
- **RPI 30:** 26.50°C (lowest)

**Interpretation:**
✅ **Significant device differences** → 4°C range between devices
⚠️ **Device calibration issues** → Each device measures differently
📊 **Analysis impact** → Must account for device effects in modeling

---

## 🎯 Key Scientific Findings

### **Environmental Relationships:**
1. **Temperature-NH3 Inverse Relationship (r = -0.627)**
   - Higher temperatures → Lower NH3 concentrations
   - Likely due to increased volatilization at higher temperatures
   - Strong predictive potential for environmental modeling

2. **Gas Sensor Clustering**
   - Oxidised and NH3 sensors both negatively correlated with temperature
   - Suggests common environmental processes affecting both measurements
   - Potential for multi-sensor environmental modeling

3. **Sound-Temperature Coupling (weak)**
   - Low-frequency sound shows modest correlation (r = 0.232)
   - May indicate thermal effects on acoustic propagation
   - Or correlation with environmental activity levels

### **Device Performance:**
1. **Measurement Consistency:** CV = 9.84% (excellent)
2. **Device Variability:** 4°C range between mean measurements
3. **Temporal Coverage:** 384 days with good continuity (better than REG dataset)

### **Data Quality Assessment:**
✅ **Strengths:**
- No missing temperature values
- Large sample size (1.5M measurements)
- Additional gas sensors not in REG dataset
- Good temporal coverage
- Low measurement variability

⚠️ **Considerations:**
- Non-normal distribution (use appropriate statistical methods)
- Device-specific biases require correction
- Heavy-tailed distribution has more extreme values

---

## 📋 Recommendations for Further Analysis

### **1. Statistical Modeling:**
- Use **non-parametric methods** due to non-normal distribution
- Apply **mixed-effects models** to account for device differences
- Consider **robust regression** methods for heavy-tailed data

### **2. Environmental Studies:**
- **Investigate NH3-temperature relationship** → Strong predictive potential
- **Model gas sensor interactions** → Multi-sensor environmental monitoring
- **Seasonal analysis** → Temporal patterns in gas concentrations

### **3. Device Calibration:**
- **Device-specific correction factors** → Improve measurement consistency
- **Cross-calibration studies** → Understand device biases
- **Quality control metrics** → Monitor device performance over time

### **4. Predictive Modeling:**
- **NH3 as primary predictor** → Strong correlation with temperature
- **Multi-sensor fusion** → Combine gas sensors for better prediction
- **Time series modeling** → Temporal patterns and forecasting

---

## 📈 Comparison with REG Dataset

| Metric | Plus Dataset | REG Dataset | Advantage |
|--------|-------------|-------------|-----------|
| **Sample Size** | 1,495,441 | 1,041,052 | Plus (+43%) |
| **Date Coverage** | 384 days | ~365 days | Plus (better continuity) |
| **Missing Values** | 0 | 0 | Equal |
| **Additional Sensors** | Gas sensors (3) | None | Plus |
| **Device Consistency** | CV = 9.84% | CV = 12.31% | Plus (better) |
| **Temperature Range** | 6-40°C | 10-676°C | Plus (no outliers) |
| **Data Gaps** | Minimal | March-April 2022 | Plus |

**Overall:** Plus dataset shows superior data quality and additional environmental monitoring capabilities.

---

*Analysis completed on 2025-09-09 using comprehensive statistical testing and visualization methods.*
