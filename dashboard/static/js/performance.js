document.addEventListener("DOMContentLoaded", () => {

    if (typeof Chart === "undefined") {
        console.error("Chart.js is not loaded.");
        return;
    }

    // =====================================
    // MODEL ACCURACY COMPARISON
    // =====================================

    const accuracyCanvas = document.getElementById("accuracyChart");

    if (accuracyCanvas) {

        new Chart(accuracyCanvas, {
            type: "bar",

            data: {
                labels: [
                    "Centralized",
                    "IID FedAvg",
                    "Non-IID FedAvg"
                ],

                datasets: [{
                    label: "Accuracy (%)",
                    data: [77.83, 74.94, 61.44],
                    backgroundColor: [
                        "#3b82f6",
                        "#10b981",
                        "#f59e0b"
                    ],
                    borderRadius: 8,
                    maxBarThickness: 65
                }]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                plugins: {
                    legend: {
                        display: false
                    }
                },

                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,

                        ticks: {
                            color: "#cbd5e1",
                            callback: value => value + "%"
                        },

                        grid: {
                            color: "rgba(255,255,255,0.07)"
                        }
                    },

                    x: {
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
    // F1 SCORE COMPARISON
    // =====================================

    const f1Canvas = document.getElementById("f1Chart");

    if (f1Canvas) {

        new Chart(f1Canvas, {
            type: "bar",

            data: {
                labels: [
                    "Centralized",
                    "IID FedAvg",
                    "Non-IID FedAvg"
                ],

                datasets: [
                    {
                        label: "Macro F1",
                        data: [0.5160, 0.4568, 0.3267],
                        backgroundColor: "#3b82f6",
                        borderRadius: 6
                    },
                    {
                        label: "Weighted F1",
                        data: [0.7711, 0.7274, 0.5315],
                        backgroundColor: "#10b981",
                        borderRadius: 6
                    }
                ]
            },

            options: {
                responsive: true,
                maintainAspectRatio: false,

                plugins: {
                    legend: {
                        position: "bottom",
                        labels: {
                            color: "#cbd5e1"
                        }
                    }
                },

                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,

                        ticks: {
                            color: "#cbd5e1"
                        },

                        grid: {
                            color: "rgba(255,255,255,0.07)"
                        }
                    },

                    x: {
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