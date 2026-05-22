// App Controller for IrisPredict Dashboard

document.addEventListener('DOMContentLoaded', () => {
    // State Variables
    let currentFeatures = {
        sepal_length: 5.1,
        sepal_width: 3.5,
        petal_length: 1.4,
        petal_width: 0.2
    };
    
    let activeModel = 'Decision Tree';
    let currentDataTablePage = 1;
    const recordsPerPage = 8;
    let filteredData = [...IRIS_DATASET];

    // DOM Elements
    const bodyEl = document.body;
    const sliderContainer = document.querySelector('.slider-group');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Model Arena Elements
    const modelPills = document.querySelectorAll('.model-pill');
    const kpiAccuracy = document.getElementById('kpi-accuracy');
    const kpiPrecision = document.getElementById('kpi-precision');
    const kpiRecall = document.getElementById('kpi-recall');
    const kpiF1 = document.getElementById('kpi-f1');
    const algoDescription = document.getElementById('algo-description');
    
    // Visualization Selector Elements
    const visSelectButtons = document.querySelectorAll('.vis-select-btn');
    const activeVisImg = document.getElementById('active-vis-image');
    
    // Prediction Elements
    const predictedSpeciesEl = document.getElementById('predicted-species-text');
    const predictedConfidenceEl = document.getElementById('predicted-confidence-text');
    const decisionPathEl = document.getElementById('decision-path-text');
    const flowerVectorContainer = document.getElementById('flower-vector-container');

    // Space Mapper Elements
    const scatterSvg = document.getElementById('scatter-svg');
    const userMarker = document.getElementById('user-marker');
    const userMarkerRing = document.getElementById('user-marker-ring');
    const compareStatsList = document.getElementById('compare-stats');

    // Data Table Elements
    const dataTableBody = document.querySelector('.data-table tbody');
    const tableSearch = document.getElementById('table-search');
    const pageInfo = document.getElementById('page-info');
    const prevPageBtn = document.getElementById('prev-page');
    const nextPageBtn = document.getElementById('next-page');

    // Flower Vector SVGs matching the predicted species
    const flowerSVGs = {
        setosa: `
            <svg class="flower-svg" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="50" cy="50" r="8" fill="#eab308" />
                <path class="animate-petal" d="M50 42C50 42 35 15 50 15C65 15 50 42 50 42Z" fill="url(#setosaGrad)" />
                <path class="animate-petal" style="transform: rotate(60deg); transform-origin: 50% 50%;" d="M50 42C50 42 35 15 50 15C65 15 50 42 50 42Z" fill="url(#setosaGrad)" />
                <path class="animate-petal" style="transform: rotate(120deg); transform-origin: 50% 50%;" d="M50 42C50 42 35 15 50 15C65 15 50 42 50 42Z" fill="url(#setosaGrad)" />
                <path class="animate-petal" style="transform: rotate(180deg); transform-origin: 50% 50%;" d="M50 42C50 42 35 15 50 15C65 15 50 42 50 42Z" fill="url(#setosaGrad)" />
                <path class="animate-petal" style="transform: rotate(240deg); transform-origin: 50% 50%;" d="M50 42C50 42 35 15 50 15C65 15 50 42 50 42Z" fill="url(#setosaGrad)" />
                <path class="animate-petal" style="transform: rotate(300deg); transform-origin: 50% 50%;" d="M50 42C50 42 35 15 50 15C65 15 50 42 50 42Z" fill="url(#setosaGrad)" />
                <defs>
                    <linearGradient id="setosaGrad" x1="50" y1="15" x2="50" y2="42" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#22d3ee" />
                        <stop offset="1" stop-color="#0891b2" stop-opacity="0.4" />
                    </linearGradient>
                </defs>
            </svg>
        `,
        versicolor: `
            <svg class="flower-svg" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="50" cy="50" r="10" fill="#eab308" />
                <path class="animate-petal" d="M50 40C50 40 25 20 50 10C75 20 50 40 50 40Z" fill="url(#versiGrad1)" />
                <path class="animate-petal" style="transform: rotate(72deg); transform-origin: 50% 50%;" d="M50 40C50 40 25 20 50 10C75 20 50 40 50 40Z" fill="url(#versiGrad2)" />
                <path class="animate-petal" style="transform: rotate(144deg); transform-origin: 50% 50%;" d="M50 40C50 40 25 20 50 10C75 20 50 40 50 40Z" fill="url(#versiGrad1)" />
                <path class="animate-petal" style="transform: rotate(216deg); transform-origin: 50% 50%;" d="M50 40C50 40 25 20 50 10C75 20 50 40 50 40Z" fill="url(#versiGrad2)" />
                <path class="animate-petal" style="transform: rotate(288deg); transform-origin: 50% 50%;" d="M50 40C50 40 25 20 50 10C75 20 50 40 50 40Z" fill="url(#versiGrad1)" />
                <defs>
                    <linearGradient id="versiGrad1" x1="50" y1="10" x2="50" y2="40" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#f472b6" />
                        <stop offset="1" stop-color="#db2777" stop-opacity="0.3" />
                    </linearGradient>
                    <linearGradient id="versiGrad2" x1="50" y1="10" x2="50" y2="40" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#ec4899" />
                        <stop offset="1" stop-color="#be185d" stop-opacity="0.3" />
                    </linearGradient>
                </defs>
            </svg>
        `,
        virginica: `
            <svg class="flower-svg" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path class="animate-stem" d="M50 50C50 50 45 75 50 90" stroke="#10b981" stroke-width="3" stroke-linecap="round" />
                <circle cx="50" cy="45" r="9" fill="#facc15" />
                <path class="animate-petal" d="M50 36C50 36 20 5 50 2C80 5 50 36 50 36Z" fill="url(#virgGrad1)" />
                <path class="animate-petal" style="transform: rotate(45deg); transform-origin: 50% 45%;" d="M50 36C50 36 20 5 50 2C80 5 50 36 50 36Z" fill="url(#virgGrad2)" />
                <path class="animate-petal" style="transform: rotate(90deg); transform-origin: 50% 45%;" d="M50 36C50 36 20 5 50 2C80 5 50 36 50 36Z" fill="url(#virgGrad1)" />
                <path class="animate-petal" style="transform: rotate(135deg); transform-origin: 50% 45%;" d="M50 36C50 36 20 5 50 2C80 5 50 36 50 36Z" fill="url(#virgGrad2)" />
                <path class="animate-petal" style="transform: rotate(180deg); transform-origin: 50% 45%;" d="M50 36C50 36 20 5 50 2C80 5 50 36 50 36Z" fill="url(#virgGrad1)" />
                <path class="animate-petal" style="transform: rotate(225deg); transform-origin: 50% 45%;" d="M50 36C50 36 20 5 50 2C80 5 50 36 50 36Z" fill="url(#virgGrad2)" />
                <path class="animate-petal" style="transform: rotate(270deg); transform-origin: 50% 45%;" d="M50 36C50 36 20 5 50 2C80 5 50 36 50 36Z" fill="url(#virgGrad1)" />
                <path class="animate-petal" style="transform: rotate(315deg); transform-origin: 50% 45%;" d="M50 36C50 36 20 5 50 2C80 5 50 36 50 36Z" fill="url(#virgGrad2)" />
                <defs>
                    <linearGradient id="virgGrad1" x1="50" y1="2" x2="50" y2="36" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#c084fc" />
                        <stop offset="1" stop-color="#7e22ce" stop-opacity="0.3" />
                    </linearGradient>
                    <linearGradient id="virgGrad2" x1="50" y1="2" x2="50" y2="36" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#a855f7" />
                        <stop offset="1" stop-color="#6b21a8" stop-opacity="0.3" />
                    </linearGradient>
                </defs>
            </svg>
        `
    };

    // Descriptions for each algorithm
    const ALGO_TEXTS = {
        'Decision Tree': 'A Decision Tree classifies data by creating a tree structure where each internal node represents a test on an attribute, each branch represents an outcome of the test, and each leaf node represents a class label. It provides excellent interpretability, partitioning the decision space into axis-aligned hyper-rectangles as visualized in the decision boundary plot.',
        'Random Forest': 'Random Forest is a robust ensemble classifier that builds an array of distinct Decision Trees during training. It merges their individual classifications (via voting) to generate a more stable, generalized prediction. By using bootstrap aggregating (bagging) and random feature selection, it successfully reduces variance and avoids overfitting, resulting in outstanding test accuracy.',
        'Logistic Regression': 'Despite the name "Regression", Logistic Regression is a classic linear model for multi-class classification using the softmax function. It computes linear decision boundaries (hyperplanes) to separate classes in feature space. This model operates under a probabilistic framework, modeling logits as linear combinations of input attributes, making it extremely fast to evaluate.'
    };

    // ==================== INTERACTIVE LIVE PREDICTOR ====================
    
    // Core Prediction Evaluator & UI Updates
    function updatePrediction() {
        // Run faithful decision tree classification
        const res = predictDecisionTree(currentFeatures);
        const species = res.prediction.toLowerCase();
        
        // Update styling state class on body (for context colors)
        bodyEl.classList.remove('state-setosa', 'state-versicolor', 'state-virginica');
        bodyEl.classList.add(`state-${species}`);

        // Update Text
        predictedSpeciesEl.textContent = res.prediction.toUpperCase();
        
        // Show probabilities
        const confidence = (Math.max(...res.probabilities) * 100).toFixed(0);
        predictedConfidenceEl.innerHTML = `Model Confidence: <span style="font-weight: 700;">${confidence}%</span>`;

        // Update Decision Path Visualizer
        // Format path arrows for clean visual display
        const pathSteps = res.path.split(' -> ');
        const pathHtml = pathSteps.map((step, idx) => {
            if (idx === 0) return `<span style="color:var(--text-muted)">${step}</span>`;
            if (idx === pathSteps.length - 1) return ` &rarr; <span style="color:var(--setosa-color); font-weight:700;">${step}</span>`;
            return ` &rarr; <span>${step}</span>`;
        }).join('');
        decisionPathEl.innerHTML = pathHtml;

        // Update Flower SVG
        flowerVectorContainer.innerHTML = flowerSVGs[species];

        // Update live stats comparing with species averages
        updateCompareStats(res.prediction);
        
        // Update Space telemetry coordinate
        updateUserTelemetryPoint();
    }

    // Handles Slider Moves
    function handleSliderInput(e) {
        const feature = e.target.dataset.feature;
        const val = parseFloat(e.target.value);
        currentFeatures[feature] = val;
        
        // Update individual numeric displays
        document.getElementById(`val-${feature.replace('_', '-')}`).textContent = val.toFixed(1) + ' cm';
        
        updatePrediction();
    }

    // Dynamic slider attachment
    const sliders = document.querySelectorAll('.custom-range');
    sliders.forEach(slider => {
        slider.addEventListener('input', handleSliderInput);
    });

    // ==================== SPACE TELEMETRY (SCATTER PLOT) ====================
    
    // Bounds & margins for custom SVG scatter plotting
    const svgWidth = 520;
    const svgHeight = 350;
    const padding = { top: 30, right: 30, bottom: 45, left: 55 };

    // Setosa, Versicolor, Virginica typical dimensions for comparison
    function updateCompareStats(predSpecies) {
        const stats = SPECIES_MEANS[predSpecies];
        if (!stats) return;

        let statsHtml = '';
        const featuresMeta = [
            { key: 'sepal_length', label: 'Sepal Length', min: 4.3, max: 7.9 },
            { key: 'sepal_width', label: 'Sepal Width', min: 2.0, max: 4.4 },
            { key: 'petal_length', label: 'Petal Length', min: 1.0, max: 6.9 },
            { key: 'petal_width', label: 'Petal Width', min: 0.1, max: 2.5 }
        ];

        featuresMeta.forEach(feat => {
            const userVal = currentFeatures[feat.key];
            const speciesAvg = stats[feat.key];
            
            // Map percentages relative to feature limits
            const range = feat.max - feat.min;
            const userPct = ((userVal - feat.min) / range) * 100;
            const avgPct = ((speciesAvg - feat.min) / range) * 100;
            
            // Set dynamic colored classes matching active species
            const speciesClass = predSpecies.toLowerCase();

            statsHtml += `
                <div class="compare-stat-item">
                    <div class="compare-stat-label">
                        <span>${feat.label}</span>
                        <span>User: <strong>${userVal.toFixed(1)}</strong> vs Avg: <strong>${speciesAvg.toFixed(1)}</strong></span>
                    </div>
                    <div class="bar-container">
                        <div class="fill-bar ${speciesClass}-color" style="width: ${userPct}%"></div>
                        <div class="avg-marker" style="left: ${avgPct}%" title="Typical ${predSpecies} Average"></div>
                    </div>
                </div>
            `;
        });

        compareStatsList.innerHTML = statsHtml;
    }

    // Scales values to SVG space coordinates
    function mapX(val) {
        // Petal Length: 1.0 cm to 7.0 cm range
        const minVal = 0.5;
        const maxVal = 7.5;
        return padding.left + ((val - minVal) / (maxVal - minVal)) * (svgWidth - padding.left - padding.right);
    }

    function mapY(val) {
        // Petal Width: 0.0 cm to 2.8 cm range
        const minVal = -0.2;
        const maxVal = 2.8;
        // SVG y=0 is at top, so subtract from height
        return svgHeight - padding.bottom - ((val - minVal) / (maxVal - minVal)) * (svgHeight - padding.top - padding.bottom);
    }

    // Plots iris dataset dots onto SVG
    function drawBaseScatterDataset() {
        const speciesColors = {
            setosa: 'var(--setosa-color)',
            versicolor: 'var(--versicolor-color)',
            virginica: 'var(--virginica-color)'
        };

        // Draw scatter dots
        let dotsHtml = '';
        IRIS_DATASET.forEach((d, idx) => {
            const cx = mapX(d.petal_length);
            const cy = mapY(d.petal_width);
            const color = speciesColors[d.species];
            
            dotsHtml += `
                <circle class="scatter-dot" cx="${cx}" cy="${cy}" r="4" 
                        fill="${color}" opacity="0.65" 
                        data-index="${idx}"
                        title="${d.species.toUpperCase()}: Petal L=${d.petal_length}, W=${d.petal_width}">
                </circle>
            `;
        });

        // Insert behind the user telemetry marker (so marker is on top)
        const baseDatasetGroup = document.getElementById('base-dataset-group');
        baseDatasetGroup.innerHTML = dotsHtml;
    }

    // Updates the position of the user's coordinate dynamically
    function updateUserTelemetryPoint() {
        const px = mapX(currentFeatures.petal_length);
        const py = mapY(currentFeatures.petal_width);

        // Update central marker
        userMarker.setAttribute('cx', px);
        userMarker.setAttribute('cy', py);

        // Update pulsing aura
        userMarkerRing.setAttribute('cx', px);
        userMarkerRing.setAttribute('cy', py);
    }

    // ==================== MODEL EVALUATION ARENA ====================

    // Update Model selection details
    function setModelArenaData(modelName) {
        activeModel = modelName;
        
        // Update Pills
        modelPills.forEach(pill => {
            if (pill.dataset.model === modelName) {
                pill.classList.add('active');
            } else {
                pill.classList.remove('active');
            }
        });

        // Fetch metrics from config
        const metrics = MODEL_METRICS[modelName];
        if (metrics) {
            kpiAccuracy.textContent = (metrics.accuracy * 100).toFixed(0) + '%';
            kpiPrecision.textContent = (metrics.precision * 100).toFixed(0) + '%';
            kpiRecall.textContent = (metrics.recall * 100).toFixed(0) + '%';
            kpiF1.textContent = (metrics.f1 * 100).toFixed(0) + '%';
        }

        // Update algorithmic text explanation
        algoDescription.textContent = ALGO_TEXTS[modelName];
    }

    // Model selection listener
    modelPills.forEach(pill => {
        pill.addEventListener('click', () => {
            setModelArenaData(pill.dataset.model);
        });
    });

    // ==================== VISUALIZATION TAB SWITCHER ====================
    visSelectButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            visSelectButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Switch image source to selected asset
            const visType = btn.dataset.vis;
            activeVisImg.src = `assets/${visType}.svg`;
            
            if (visType === 'decision_boundaries') {
                activeVisImg.classList.add('vis-img-boundary');
            } else {
                activeVisImg.classList.remove('vis-img-boundary');
            }
        });
    });

    // ==================== DATA EXPLORER TABLE ====================

    function renderExplorerTable() {
        const start = (currentDataTablePage - 1) * recordsPerPage;
        const end = start + recordsPerPage;
        const pageRecords = filteredData.slice(start, end);

        let rowsHtml = '';
        if (pageRecords.length === 0) {
            rowsHtml = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted);">No records found matching search query.</td></tr>`;
        } else {
            pageRecords.forEach(d => {
                const badgeClass = `${d.species}-color`;
                rowsHtml += `
                    <tr>
                        <td><strong>${d.sepal_length.toFixed(1)}</strong> cm</td>
                        <td><strong>${d.sepal_width.toFixed(1)}</strong> cm</td>
                        <td><strong>${d.petal_length.toFixed(1)}</strong> cm</td>
                        <td><strong>${d.petal_width.toFixed(1)}</strong> cm</td>
                        <td><span class="legend-color ${badgeClass}" style="display: inline-block; vertical-align: middle; margin-right: 6px;"></span><span style="text-transform: capitalize; font-weight:600;">${d.species}</span></td>
                    </tr>
                `;
            });
        }

        dataTableBody.innerHTML = rowsHtml;
        
        // Update pagination descriptors
        pageInfo.textContent = `Page ${currentDataTablePage} of ${Math.ceil(filteredData.length / recordsPerPage) || 1}`;
        prevPageBtn.disabled = currentDataTablePage === 1;
        nextPageBtn.disabled = currentDataTablePage >= Math.ceil(filteredData.length / recordsPerPage);
    }

    // Handles Search Filtering
    tableSearch.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        filteredData = IRIS_DATASET.filter(d => d.species.toLowerCase().includes(query));
        currentDataTablePage = 1; // Reset to page 1
        renderExplorerTable();
    });

    // Pagination events
    prevPageBtn.addEventListener('click', () => {
        if (currentDataTablePage > 1) {
            currentDataTablePage--;
            renderExplorerTable();
        }
    });

    nextPageBtn.addEventListener('click', () => {
        if (currentDataTablePage < Math.ceil(filteredData.length / recordsPerPage)) {
            currentDataTablePage++;
            renderExplorerTable();
        }
    });

    // ==================== TABS INTERACTION ====================
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            document.getElementById(targetTab).classList.add('active');
            
            // Special initialization adjustments if needed when tab is revealed
            if (targetTab === 'space-explorer-tab') {
                // Ensure scatter updates coordinates correctly
                updateUserTelemetryPoint();
            }
        });
    });

    // ==================== APPLICATION INITIALIZATION ====================
    function init() {
        // Build base scatter layout coordinates
        drawBaseScatterDataset();
        
        // Trigger initial predictions
        updatePrediction();
        
        // Set initial Model Arena display values
        setModelArenaData('Decision Tree');
        
        // Render search data
        renderExplorerTable();
    }

    init();
});
