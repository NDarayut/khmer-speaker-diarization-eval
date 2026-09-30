// Hash router: #/ = recordings, #/r/<id> = comparison, #/review = spot check, #/dataset.
import { showLibrary, hideLibrary } from "./library.js";
import { showCompare, hideCompare } from "./compare.js";
import { showReview, hideReview } from "./review.js";
import { showDataset, hideDataset } from "./dataset.js";
import { $, $$, toast } from "./util.js";

const VIEWS = {
  library: [showLibrary, hideLibrary],
  compare: [showCompare, hideCompare],
  review: [showReview, hideReview],
  dataset: [showDataset, hideDataset],
};

async function route() {
  const m = location.hash.match(/^#\/r\/([A-Za-z0-9._-]+)/);
  const view = m ? "compare" : location.hash.startsWith("#/review") ? "review"
    : location.hash.startsWith("#/dataset") ? "dataset" : "library";
  for (const [name, [, hide]] of Object.entries(VIEWS)) if (name !== view) hide();
  $$("#tabs a").forEach((a) => a.setAttribute("aria-current", String(a.dataset.view === view)));
  try {
    await VIEWS[view][0](m?.[1]);
    if (view === "library") document.title = "Diarization Bench";
  } catch (err) {
    console.error(err);
    toast(err.message);
  }
}

$("#help-btn").addEventListener("click", () => ($("#help").hidden = false));
$("#help").addEventListener("click", (e) => {
  if (e.target.id === "help" || e.target.closest("[data-close]")) $("#help").hidden = true;
});
document.addEventListener("keydown", (e) => {
  if (e.key === "?" && !/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) $("#help").hidden = false;
  if (e.key === "Escape" && !$("#help").hidden) $("#help").hidden = true;
});
window.addEventListener("hashchange", route);
route();
