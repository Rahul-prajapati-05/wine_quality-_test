/**
 * VinoPredict AI - Frontend Application Controller
 * Handles slider two-way binding, preset loading, real-time predictions,
 * and interactive EDA tab navigation.
 */

document.addEventListener('DOMContentLoaded', () => {

    // Feature key definitions
    const FEATURE_KEYS = [
        'alcohol',
        'volatile_acidity',
        'sulphates',
        'citric_acid',
        'total_sulfur_dioxide',
        'free_sulfur_dioxide',
        'fixed_acidity',
        'chlorides',
        'density',
        'pH',
        'residual_sugar'
    ];

    // Presets Cache
    const PRESETS = {
        premium_reserve: {
            alcohol: 12.8,
            volatile_acidity: 0.28,
            sulphates: 0.88,
            citric_acid: 0.48,
            total_sulfur_dioxide: 38.0,
            free_sulfur_dioxide: 18.0,
            fixed_acidity: 8.5,
            chlorides: 0.055,
            density: 0.9958,
            pH: 3.32,
            residual_sugar: 2.2
        },
        commercial_table: {
            alcohol: 10.2,
            volatile_acidity: 0.58,
            sulphates: 0.62,
            citric_acid: 0.15,
            total_sulfur_dioxide: 46.0,
            free_sulfur_dioxide: 14.0,
            fixed_acidity: 7.4,
            chlorides: 0.082,
            density: 0.9968,
            pH: 3.38,
            residual_sugar: 2.1
        },
        defective_batch: {
            alcohol: 9.1,
            volatile_acidity: 0.98,
            sulphates: 0.44,
            citric_acid: 0.02,
            total_sulfur_dioxide: 28.0,
            free_sulfur_dioxide: 8.0,
            fixed_acidity: 7.2,
            chlorides: 0.115,
            density: 0.9972,
            pH: 3.52,
            residual_sugar: 1.8
        },
        dataset_mean: {
            alcohol: 10.42,
            volatile_acidity: 0.53,
            sulphates: 0.66,
            citric_acid: 0.27,
            total_sulfur_dioxide: 46.47,
            free_sulfur_dioxide: 15.87,
            fixed_acidity: 8.32,
            chlorides: 0.087,
            density: 0.9967,
            pH: 3.31,
            residual_sugar: 2.54
        }
    };

    // =========================================================================
    // 1. SLIDER BINDINGS
    // =========================================================================
    FEATURE_KEYS.forEach(key => {
        const slider = document.getElementById(`${key}-range`);
        const display = document.getElementById(`${key}-val`);

        if (slider && display) {
            slider.addEventListener('input', () => {
                display.textContent = slider.value;
            });
        }
    });

    // =========================================================================
    // 2. PRESET BUTTONS
    // =========================================================================
    const presetBtns = document.querySelectorAll('.preset-btn');
    presetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            presetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const presetKey = btn.dataset.preset;
            if (PRESETS[presetKey]) {
                applyPreset(PRESETS[presetKey]);
                // Automatically run prediction on preset switch
                runPrediction();
            }
        });
    });

    function applyPreset(values) {
        Object.keys(values).forEach(key => {
            const slider = document.getElementById(`${key}-range`);
            const display = document.getElementById(`${key}-val`);
            if (slider && display) {
                slider.value = values[key];
                display.textContent = values[key];
            }
        });
    }

    // =========================================================================
    // 3. PREDICTION TRIGGER & API CALL
    // =========================================================================
    const submitBtn = document.getElementById('predict-submit-btn');
    const spinner = document.getElementById('predict-spinner');
    const btnText = document.getElementById('predict-btn-text');

    if (submitBtn) {
        submitBtn.addEventListener('click', runPrediction);
    }

    async function runPrediction() {
        // Collect current values
        const payload = {};
        FEATURE_KEYS.forEach(key => {
            const slider = document.getElementById(`${key}-range`);
            if (slider) {
                payload[key.replace('_', ' ')] = parseFloat(slider.value);
            }
        });

        // Set UI loading state
        if (spinner) spinner.style.display = 'inline-block';
        if (btnText) btnText.textContent = 'Classifying Assays...';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`Server returned HTTP ${response.status}`);
            }

            const result = await response.json();
            updateResultCard(result);
        } catch (err) {
            console.warn('[VinoPredict] Direct API offline or fetch failed, falling back to local client heuristic:', err);
            // Client-side fallback heuristic
            const fallbackResult = computeClientHeuristic(payload);
            updateResultCard(fallbackResult);
        } finally {
            if (spinner) spinner.style.display = 'none';
            if (btnText) btnText.textContent = 'Classify Wine Quality Tier';
        }
    }

    // =========================================================================
    // 4. CLIENT HEURISTIC FALLBACK
    // =========================================================================
    function computeClientHeuristic(p) {
        const alc = p['alcohol'] || 10.4;
        const va = p['volatile acidity'] || 0.52;
        const so4 = p['sulphates'] || 0.65;

        let predClass = 1;
        let probLow = 0.08;
        let probMed = 0.80;
        let probHigh = 0.12;

        if (va >= 0.75 || (va >= 0.60 && alc < 9.6)) {
            predClass = 0; // Low
            probLow = 0.74;
            probMed = 0.22;
            probHigh = 0.04;
        } else if (alc >= 11.4 && va <= 0.45 && so4 >= 0.65) {
            predClass = 2; // High
            probLow = 0.03;
            probMed = 0.15;
            probHigh = 0.82;
        }

        const tierInfo = {
            0: {
                tier_name: "Low Quality",
                tier_label: "Low (Grade 3–4)",
                tier_badge: "Defective / Budget Tier",
                tier_color: "#e63946",
                stars: 2,
                desc: "Exhibits sensory flaws such as high volatile acidity (vinegar note) or deficient alcohol balance. Suitable strictly for industrial vinegar or low-grade bulk blending."
            },
            1: {
                tier_name: "Medium Quality",
                tier_label: "Medium (Grade 5–6)",
                tier_badge: "Standard Commercial Wine",
                tier_color: "#e3a857",
                stars: 3,
                desc: "Sound, drinkable commercial table wine meeting standard market quality specifications. Well-balanced acidity and moderate alcohol content."
            },
            2: {
                tier_name: "High Quality",
                tier_label: "High (Grade 7–8)",
                tier_badge: "Premium / Reserve Selection",
                tier_color: "#2a9d8f",
                stars: 5,
                desc: "Exceptional wine with optimal alcohol body, pristine volatile acidity control, and strong antioxidant sulphate protection. Premium export quality."
            }
        };

        const t = tierInfo[predClass];
        const estScore = (probLow * 3.5 + probMed * 5.5 + probHigh * 7.4).toFixed(2);

        return {
            predicted_class: predClass,
            tier_name: t.tier_name,
            tier_label: t.tier_label,
            tier_badge: t.tier_badge,
            tier_color: t.tier_color,
            stars: t.stars,
            description: t.desc,
            confidence: predClass === 2 ? probHigh : (predClass === 0 ? probLow : probMed),
            probabilities: {
                "Low (3-4)": probLow,
                "Medium (5-6)": probMed,
                "High (7-8)": probHigh
            },
            estimated_score: estScore,
            oenological_profile: {
                alcohol_assessment: alc >= 11.5 ? "High (Rich Body)" : (alc < 9.5 ? "Low (Thin Body)" : "Moderate Commercial Body"),
                volatile_acidity_assessment: va > 0.65 ? "Defective (Vinegar)" : (va < 0.45 ? "Clean & Aromatic" : "Acceptable"),
                sulphates_assessment: so4 >= 0.70 ? "Optimal Preservation" : "Moderate Protection"
            },
            actionable_tips: [
                va > 0.60 ? "⚠️ Volatile Acidity is elevated (>0.60 g/dm³). Minimize bacterial oxidation." : "✅ Volatile acidity is safely below sensory defect threshold.",
                alc >= 11.5 ? "✅ Alcohol concentration provides robust structure and lingering finish." : "ℹ️ Harvesting at higher Brix maturity would elevate richness."
            ]
        };
    }

    // =========================================================================
    // 5. UPDATE RESULT CARD DOM
    // =========================================================================
    function updateResultCard(data) {
        // Star string
        const starCount = data.stars || 3;
        const starStr = '⭐'.repeat(starCount);
        const starsEl = document.getElementById('tier-stars-container');
        if (starsEl) starsEl.textContent = starStr;

        // Title and description
        const titleEl = document.getElementById('tier-title');
        if (titleEl) titleEl.textContent = data.tier_name;

        const badgeEl = document.getElementById('tier-pill-badge');
        if (badgeEl) {
            badgeEl.textContent = data.tier_badge;
            if (data.tier_color) {
                badgeEl.style.borderColor = data.tier_color;
                badgeEl.style.color = data.tier_color;
            }
        }

        const descEl = document.getElementById('tier-description');
        if (descEl) descEl.textContent = data.description;

        // Estimated score gauge
        const scoreValEl = document.getElementById('estimated-score-val');
        if (scoreValEl) scoreValEl.textContent = `${data.estimated_score} / 10`;

        const scoreBarEl = document.getElementById('score-gauge-bar');
        if (scoreBarEl) {
            // Map 3.0 -> 8.5 range to 0% -> 100%
            const pct = Math.min(100, Math.max(10, ((data.estimated_score - 3.0) / 5.5) * 100));
            scoreBarEl.style.width = `${pct}%`;
        }

        // Probability bars
        const probs = data.probabilities || {};
        const pHigh = Math.round((probs["High (7-8)"] || 0) * 100);
        const pMed = Math.round((probs["Medium (5-6)"] || 0) * 100);
        const pLow = Math.round((probs["Low (3-4)"] || 0) * 100);

        updateProbRow('high', pHigh);
        updateProbRow('med', pMed);
        updateProbRow('low', pLow);

        // Oenological Profile
        const prof = data.oenological_profile || {};
        updateDiagnosis('diag-alcohol', prof.alcohol_assessment);
        updateDiagnosis('diag-va', prof.volatile_acidity_assessment);
        updateDiagnosis('diag-so4', prof.sulphates_assessment);

        // Actionable Tips
        const tipsList = document.getElementById('actionable-tips-list');
        if (tipsList && Array.isArray(data.actionable_tips)) {
            tipsList.innerHTML = '';
            data.actionable_tips.forEach(tip => {
                const li = document.createElement('li');
                li.textContent = tip;
                tipsList.appendChild(li);
            });
        }
    }

    function updateProbRow(tierKey, val) {
        const textEl = document.getElementById(`prob-val-${tierKey}`);
        const barEl = document.getElementById(`prob-bar-${tierKey}`);
        if (textEl) textEl.textContent = `${val}%`;
        if (barEl) barEl.style.width = `${val}%`;
    }

    function updateDiagnosis(id, text) {
        const el = document.getElementById(id);
        if (!el || !text) return;
        el.textContent = text;
        el.className = 'diag-status';

        if (text.includes('High') || text.includes('Clean') || text.includes('Optimal') || text.includes('Excellent')) {
            el.classList.add('status-good');
        } else if (text.includes('Defective') || text.includes('Vinegar') || text.includes('Thin')) {
            el.classList.add('status-bad');
        } else {
            el.classList.add('status-warning');
        }
    }

    // =========================================================================
    // 6. EDA TABS CONTROLLER
    // =========================================================================
    const edaTabs = document.querySelectorAll('.eda-tab');
    const edaPanes = document.querySelectorAll('.eda-pane');

    edaTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.target;

            edaTabs.forEach(t => t.classList.remove('active'));
            edaPanes.forEach(p => p.classList.remove('active'));

            tab.classList.add('active');
            const activePane = document.getElementById(targetId);
            if (activePane) {
                activePane.classList.add('active');
            }
        });
    });

    // Check backend health on startup
    fetch('/api/health')
        .then(res => res.json())
        .then(data => {
            const dot = document.getElementById('api-status-dot');
            const txt = document.getElementById('api-status-text');
            if (dot) dot.classList.add('live');
            if (txt) txt.textContent = 'API Live (RF Active)';
        })
        .catch(() => {
            const dot = document.getElementById('api-status-dot');
            const txt = document.getElementById('api-status-text');
            if (dot) dot.style.backgroundColor = '#f59e0b';
            if (txt) txt.textContent = 'Client Heuristic Mode';
        });

    // Initialize with preset 1
    applyPreset(PRESETS.premium_reserve);

});
