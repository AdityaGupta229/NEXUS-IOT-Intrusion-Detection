document.addEventListener("DOMContentLoaded", () => {

    let trafficChart = null;
    let classChart = null;

    // Load latest saved analysis
    function getSavedAnalysis() {
        try {
            return JSON.parse(
                localStorage.getItem("nexusLastAnalysis")
            );
        } catch (error) {
            console.error("Unable to load analysis:", error);
            return null;
        }
    }

    const data = getSavedAnalysis();

    if (!data || !data.summary) {
        console.log("No previous analysis available.");
        return;
    }

    // =====================================
    // TRAFFIC DISTRIBUTION CHART
    // =====================================

    const trafficCanvas = document.getElementById(
        "trafficDistributionChart"
    );

    if (trafficCanvas && typeof Chart !== "undefined") {

        trafficChart = new Chart(trafficCanvas, {
            type: "doughnut",

            data: {
                labels: ["Normal Traffic", "Attack Traffic"],

                datasets: [{
                    data: [
                        data.summary.normal_records,
                        data.summary.attack_records
                    ],

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
    // ATTACK CATEGORY CHART
    // =====================================

    const classCanvas = document.getElementById(
        "classDistributionChart"
    );

    if (classCanvas && typeof Chart !== "undefined") {

        const distribution =
            data.summary.attack_distribution || {};

        const attackClasses = Object.entries(distribution)
            .filter(([label]) => {
                const normalized = label
                    .toLowerCase()
                    .replace(/[^a-z0-9]/g, "");

                return !["benign", "benigntraffic"].includes(normalized);
            })
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        classChart = new Chart(classCanvas, {
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

});