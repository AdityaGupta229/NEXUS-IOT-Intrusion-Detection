document.addEventListener("DOMContentLoaded", () => {

    const totalRecords = document.getElementById("totalRecords");
    const normalRecords = document.getElementById("normalRecords");
    const attackRecords = document.getElementById("attackRecords");

    function loadOverview() {

        let data = null;

        try {
            data = JSON.parse(
                localStorage.getItem("nexusLastAnalysis")
            );
        } catch (error) {
            console.error("Unable to load overview data:", error);
        }

        if (!data || !data.summary) return;

        if (totalRecords) {
            totalRecords.textContent =
                data.summary.total_records.toLocaleString("en-IN");
        }

        if (normalRecords) {
            normalRecords.textContent =
                data.summary.normal_records.toLocaleString("en-IN");
        }

        if (attackRecords) {
            attackRecords.textContent =
                data.summary.attack_records.toLocaleString("en-IN");
        }
    }

    loadOverview();

});