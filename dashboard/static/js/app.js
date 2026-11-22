const analyzeBtn = document.getElementById("analyzeBtn");
const fileInput = document.getElementById("file");
const modelSelect = document.getElementById("model");
const message = document.getElementById("message");

let trafficChart = null;
let classChart = null;

// =====================================
// LOAD SAVED ANALYSIS
// =====================================

function getSavedAnalysis() {
    try {
        return JSON.parse(
            localStorage.getItem("nexusLastAnalysis")
        );
    } catch (error) {
        console.error("Unable to load saved analysis:", error);
        return null;
    }
}

// =====================================
// CHART 1: NORMAL VS ATTACK
// =====================================

function updateTrafficChart(normal, attack) {
    const ctx = document.getElementById("trafficDistributionChart");

    if (!ctx || typeof Chart === "undefined") return;

    if (trafficChart) trafficChart.destroy();

    trafficChart = new Chart(ctx, {
        type: "doughnut",
        data: {
            labels: ["Normal Traffic", "Attack Traffic"],
            datasets: [{
                data: [normal, attack],
                backgroundColor: ["#10b981", "#ef4444"],
                borderColor: ["#0f172a", "#0f172a"],
                borderWidth: 3,
                hoverOffset: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: "bottom",
                    labels: {
                        color: "#cbd5e1",
                        padding: 20
                    }
                }
            },
            cutout: "68%"
        }
    });
}

// =====================================
// CHART 2: ATTACK CLASS DISTRIBUTION
// =====================================

function updateClassChart(distribution) {
    const ctx = document.getElementById("classDistributionChart");

    if (!ctx || typeof Chart === "undefined") return;

    if (classChart) classChart.destroy();

    const attackClasses = Object.entries(distribution || {})
        .filter(([label]) => {
            const normalized = label.toLowerCase().replace(/[^a-z0-9]/g, "");
            return !["benign", "benigntraffic"].includes(normalized);
        })
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);

    classChart = new Chart(ctx, {
        type: "bar",
        data: {
            labels: attackClasses.map(item => item[0]),
            datasets: [{
                label: "Predicted Records",
                data: attackClasses.map(item => item[1]),
                backgroundColor: "#3b82f6",
                borderRadius: 7,
                maxBarThickness: 35
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            indexAxis: "y",
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    beginAtZero: true,
                    ticks: {
                        color: "#cbd5e1",
                        precision: 0
                    },
                    grid: {
                        color: "rgba(255,255,255,0.07)"
                    }
                },
                y: {
                    ticks: {
                        color: "#cbd5e1"
                    },
                    grid: {
                        display: false
                    }
                }
            }
        }
    });
}

// =====================================
// OVERVIEW STATISTICS
// =====================================

function updateStatistics(data) {
    const total = document.getElementById("totalRecords");
    const normal = document.getElementById("normalRecords");
    const attack = document.getElementById("attackRecords");

    if (total) total.textContent = data.summary.total_records;
    if (normal) normal.textContent = data.summary.normal_records;
    if (attack) attack.textContent = data.summary.attack_records;
}

// =====================================
// DETECTION RESULTS TABLE
// =====================================

function updateResultsTable(data) {
    const resultsBody = document.getElementById("resultsBody");

    if (!resultsBody) return;

    resultsBody.innerHTML = "";

    const modelLabel = document.getElementById("resultModel");
    const resultCount = document.getElementById("resultCount");

    if (modelLabel) {
        modelLabel.textContent = "Model: " + data.model;
    }

    if (resultCount) {
        resultCount.textContent = data.results.length + " RECORDS";
    }

    data.results.forEach(result => {
        const row = document.createElement("tr");

        const recordCell = document.createElement("td");
        recordCell.textContent = result.record;
        row.appendChild(recordCell);

        const predictionCell = document.createElement("td");
        predictionCell.textContent = result.prediction;
        row.appendChild(predictionCell);

        const statusCell = document.createElement("td");
        const statusBadge = document.createElement("span");

        const isNormal = result.status.toLowerCase() === "normal";

        statusBadge.className = isNormal
            ? "status-badge normal"
            : "status-badge attack";

        statusBadge.textContent = result.status;
        statusCell.appendChild(statusBadge);
        row.appendChild(statusCell);

        const confidenceCell = document.createElement("td");
        const confidenceWrapper = document.createElement("div");
        confidenceWrapper.className = "confidence-wrapper";

        const confidenceBar = document.createElement("div");
        confidenceBar.className = "confidence-bar";

        const confidenceFill = document.createElement("div");
        confidenceFill.className = "confidence-fill";

        const confidence = Math.min(
            100,
            Math.max(0, Number(result.confidence) || 0)
        );

        confidenceFill.style.width = confidence + "%";

        const confidenceText = document.createElement("span");
        confidenceText.className = "confidence-text";
        confidenceText.textContent = confidence.toFixed(2) + "%";

        confidenceBar.appendChild(confidenceFill);
        confidenceWrapper.appendChild(confidenceBar);
        confidenceWrapper.appendChild(confidenceText);

        confidenceCell.appendChild(confidenceWrapper);
        row.appendChild(confidenceCell);

        resultsBody.appendChild(row);
    });
}

// =====================================
// UPDATE CURRENT PAGE
// =====================================

function displaySavedAnalysis() {
    const data = getSavedAnalysis();

    if (!data || !data.summary || !Array.isArray(data.results)) return;

    updateStatistics(data);
    updateResultsTable(data);

    updateTrafficChart(
        data.summary.normal_records,
        data.summary.attack_records
    );

    updateClassChart(data.summary.attack_distribution);
}

displaySavedAnalysis();

// =====================================
// ANALYZE CSV
// =====================================

if (analyzeBtn && fileInput && modelSelect && message) {

    analyzeBtn.addEventListener("click", async () => {
        const file = fileInput.files[0];

        if (!file) {
            message.textContent = "Please select a CSV file first.";
            return;
        }

        if (!file.name.toLowerCase().endsWith(".csv")) {
            message.textContent = "Only CSV files are allowed.";
            return;
        }

        const formData = new FormData();
        formData.append("file", file);
        formData.append("model", modelSelect.value);

        analyzeBtn.disabled = true;
        analyzeBtn.textContent = "Analyzing...";
        message.textContent = "Processing network traffic...";

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Prediction failed.");
            }

            // Save complete analysis for other pages
            localStorage.setItem("nexusLastAnalysis", JSON.stringify(data));

displaySavedAnalysis();

const resultsSection = document.getElementById("analysisResults");

if (resultsSection) {
    resultsSection.hidden = false;
    resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}

message.textContent = "Analysis completed successfully using " + data.model + ".";

        } catch (error) {
            message.textContent = error.message;

        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = "Analyze Network Traffic";
        }
    });
}