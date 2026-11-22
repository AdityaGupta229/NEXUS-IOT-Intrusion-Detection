document.addEventListener("DOMContentLoaded", () => {

    const sidebar = document.getElementById("sidebar");
    const menuToggle = document.getElementById("menuToggle");

    if (!sidebar || !menuToggle) return;

    // Toggle mobile sidebar
    menuToggle.addEventListener("click", () => {
        sidebar.classList.toggle("open");
    });

    // Close sidebar after selecting a page
    const navLinks = sidebar.querySelectorAll(".nav-item");

    navLinks.forEach(link => {
        link.addEventListener("click", () => {
            sidebar.classList.remove("open");
        });
    });

    // Close sidebar with Escape key
    document.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
            sidebar.classList.remove("open");
        }
    });

});