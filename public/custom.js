(() => {
  const IMAGE_NAME = "langgraph-architecture.png";

  function isLangGraphImage(element) {
    if (!(element instanceof HTMLImageElement)) {
      return false;
    }

    const src = element.getAttribute("src") || "";

    return src.includes(IMAGE_NAME);
  }

  document.addEventListener(
    "click",
    (event) => {
      const image = event.target.closest?.("img");

      if (!isLangGraphImage(image)) {
        return;
      }

      event.preventDefault();
      event.stopPropagation();

      const url = new URL(
        "/public/langgraph-architecture.png",
        window.location.origin
      );

      window.open(url.href, "_blank", "noopener,noreferrer");
    },
    true
  );

  function decorateImages() {
    document.querySelectorAll("img").forEach((image) => {
      if (!isLangGraphImage(image)) {
        return;
      }

      image.style.cursor = "zoom-in";
      image.title = "Click to open full-size diagram";
      image.setAttribute("role", "button");
      image.setAttribute("tabindex", "0");
    });
  }

  document.addEventListener("keydown", (event) => {
    if (
      (event.key === "Enter" || event.key === " ") &&
      isLangGraphImage(event.target)
    ) {
      event.preventDefault();

      window.open(
        new URL(
          "/public/langgraph-architecture.png",
          window.location.origin
        ).href,
        "_blank",
        "noopener,noreferrer"
      );
    }
  });

  decorateImages();

  const observer = new MutationObserver(decorateImages);

  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
})();
