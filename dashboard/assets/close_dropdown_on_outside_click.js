console.log("Dropdown close script loaded");

document.addEventListener("click", function (event) {
  const openDetails = document.querySelectorAll("details[open]");

  openDetails.forEach((details) => {
    if (!details.contains(event.target)) {
      details.removeAttribute("open");
    }
  });
});
