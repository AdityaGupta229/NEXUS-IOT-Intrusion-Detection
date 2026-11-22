document.addEventListener("DOMContentLoaded", () => {

    const resultsBody = document.getElementById("resultsBody");
    const resultModel = document.getElementById("resultModel");
    const resultCount = document.getElementById("resultCount");

    if (!resultsBody) return;

    // Load saved analysis
    let data = null;

    try {
        data = JSON.parse(
            localStorage.getItem("nexusLastAnalysis")
        );
    } catch (error) {
        console.error("Unable to load saved results:", error);
    }

    // No previous analysis
    if (!data || !Array.isArray(data.results)) {
        resultsBody.innerHTML = `
            <tr>
                <td colspan="4" class="empty-state">
                    No detection results available.
                    Analyze a CSV dataset first.
                </td>
            </tr>
        `;
        return;
    }

    // Update page information
    if (resultModel) {
        resultModel.textContent = "Model: " + data.model;
    }

    if (resultCount) {
        resultCount.textContent =
            data.results.length + " RECORDS";
    }

    // Populate prediction table
    resultsBody.innerHTML = "";

    data.results.forEach(result => {

        const row = document.createElement("tr");

        // Record number
        const recordCell = document.createElement("td");
        recordCell.textContent = result.record;
        row.appendChild(recordCell);

        // Predicted class
        const predictionCell = document.createElement("td");
        predictionCell.textContent = result.prediction;
        row.appendChild(predictionCell);

        // Detection status
        const statusCell = document.createElement("td");
        const statusBadge = document.createElement("span");

        const isNormal =
            result.status.toLowerCase() === "normal";

        statusBadge.className = isNormal
            ? "status-badge normal"
            : "status-badge attack";

        statusBadge.textContent = result.status;

        statusCell.appendChild(statusBadge);
        row.appendChild(statusCell);

        // Confidence estimate
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
        confidenceText.textContent =
            confidence.toFixed(2) + "%";

        confidenceBar.appendChild(confidenceFill);

        confidenceWrapper.appendChild(confidenceBar);
        confidenceWrapper.appendChild(confidenceText);

        confidenceCell.appendChild(confidenceWrapper);
        row.appendChild(confidenceCell);

        resultsBody.appendChild(row);

    });

});