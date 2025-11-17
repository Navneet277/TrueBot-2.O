document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll("[data-loading]");
  forms.forEach((form) => {
    form.addEventListener("submit", () => {
      const button = form.querySelector("button[type='submit']");
      if (button) {
        button.disabled = true;
        button.innerText = "Processing...";
      }
    });
  });
});


