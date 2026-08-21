
(function () {
  const DISCLAIMER = "A public demo using a small, curated subset of Princess Cruises and Royal Caribbean reference documents. Not an exhaustive production solution.";

  console.log("[custom.js] loaded");

  function addDisclaimer() {
    if (document.getElementById("cruise-demo-disclaimer")) {
      return;
    }

    const p = document.createElement("p");
    p.id = "cruise-demo-disclaimer";
    p.textContent = DISCLAIMER;
    p.style.cssText =
      "text-align: center; color: #94a3b8; font-size: 0.9rem; margin: 0.5rem auto 1.25rem auto; max-width: 32rem; padding: 0 1rem; line-height: 1.4;";

    // Try to place the text just below the cruise ship logo.
    const logo = document.querySelector('img[src*="cruise-ship"]');
    if (logo && logo.parentElement) {
      logo.parentElement.insertBefore(p, logo.nextSibling);
      console.log("[custom.js] disclaimer inserted under logo");
      return;
    }

    // Fall back to inserting just above the message composer.
    const composer = document.querySelector('textarea[placeholder*="Type your message"]');
    if (composer && composer.parentElement) {
      composer.parentElement.insertBefore(p, composer);
      console.log("[custom.js] disclaimer inserted above composer");
      return;
    }

    // Last resort: prepend to body so it is at least visible on the landing page.
    document.body.prepend(p);
    console.log("[custom.js] disclaimer prepended to body");
  }

  function tryAddDisclaimer() {
    if (document.getElementById("cruise-demo-disclaimer")) {
      return;
    }
    addDisclaimer();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", tryAddDisclaimer);
  } else {
    tryAddDisclaimer();
  }

  const observer = new MutationObserver(tryAddDisclaimer);
  observer.observe(document.body, { childList: true, subtree: true });

  // Chainlit's landing screen renders after load, so keep checking briefly.
  let retries = 0;
  const interval = setInterval(() => {
    tryAddDisclaimer();
    if (document.getElementById("cruise-demo-disclaimer") || ++retries > 10) {
      clearInterval(interval);
    }
  }, 500);
})();
