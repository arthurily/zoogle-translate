const LETTERS = Array.from("ABCDEFGHIJKLMNOPQRSTUVWXYZ");

const IMAGE_SIZE = 24;
const INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE;

const CONV_FILTERS = 14;
const KERNEL_SIZE = 5;
const STRIDE = 2;
const CONV_OUT = Math.floor((IMAGE_SIZE - KERNEL_SIZE) / STRIDE) + 1;
const FLAT_DIM = CONV_FILTERS * CONV_OUT * CONV_OUT;

const MIN_SAMPLES_PER_LETTER = 5;
const MIN_READY_LETTERS_FOR_TRAIN = 4;
const TRAIN_AUGMENTATIONS = 4;
const VALIDATION_SPLIT = 0.2;

const EPOCHS = 22;
const BATCH_SIZE = 20;
const LEARNING_RATE = 0.0098;
const INFERENCE_TEMPERATURE = 0.82;

const STORAGE_KEY = "zoogle-translate-v1";
const THEME_KEY = "zoogle-theme-v1";
const LANGUAGES_API_PATH = "/api/languages";
const SERVER_SAVE_DEBOUNCE_MS = 260;
const THEME_SEQUENCE = ["light", "dark", "extraterrestrial"];
const THEME_LABEL = {
  light: "Light",
  dark: "Dark",
  extraterrestrial: "Extraterrestrial",
};

let serverSaveTimer = null;
let serverSaveInFlight = false;
let serverSavePending = false;
let languageApiAvailability = "unknown";

const LETTER_TO_INDEX = Object.fromEntries(LETTERS.map((ch, i) => [ch, i]));

const state = {
  languages: {},
  activeLanguageId: null,
  languageName: "",
  samplesByLetter: null,
  activeLetterIndex: 0,

  model: null,
  isTraining: false,
  metrics: {
    valAcc: null,
    valLoss: null,
    trainedAt: null,
  },

  setupPad: null,
  testPad: null,

  posteriorRows: [],
  bayesRows: [],

  activeView: "translate",
  translateTokens: [],
  lastTranslation: [],
  popoverActiveIndex: -1,
  popoverLocked: false,
  popoverHideTimer: null,
};

const el = {
  body: document.body,
  navButtons: Array.from(document.querySelectorAll(".nav-btn")),
  views: {
    setup: document.getElementById("view-setup"),
    translate: document.getElementById("view-translate"),
    test: document.getElementById("view-test"),
    about: document.getElementById("view-about"),
  },

  themeToggle: document.getElementById("theme-toggle"),

  languageSelect: document.getElementById("language-select"),
  newLanguageBtn: document.getElementById("new-language-btn"),
  deleteLanguageBtn: document.getElementById("delete-language-btn"),

  statLanguage: document.getElementById("stat-language"),
  statTotal: document.getElementById("stat-total"),
  statReady: document.getElementById("stat-ready"),
  statModel: document.getElementById("stat-model"),

  languageNameInput: document.getElementById("language-name-input"),
  translateLanguageName: document.getElementById("translate-language-name"),

  totalSamples: document.getElementById("total-samples"),
  readyLetters: document.getElementById("ready-letters"),
  activeLetterCount: document.getElementById("active-letter-count"),
  activeLetterChip: document.getElementById("active-letter-chip"),
  setupStatus: document.getElementById("setup-status"),
  letterGrid: document.getElementById("letter-grid"),

  guideCanvas: document.getElementById("guide-canvas"),
  datasetCanvas: document.getElementById("dataset-canvas"),
  saveSampleBtn: document.getElementById("save-sample-btn"),
  clearSampleBtn: document.getElementById("clear-sample-btn"),
  prevLetterBtn: document.getElementById("prev-letter-btn"),
  nextLetterBtn: document.getElementById("next-letter-btn"),

  resetLetterBtn: document.getElementById("reset-letter-btn"),
  resetDatasetBtn: document.getElementById("reset-dataset-btn"),
  exportDatasetBtn: document.getElementById("export-dataset-btn"),
  importDatasetBtn: document.getElementById("import-dataset-btn"),
  importDatasetInput: document.getElementById("import-dataset-input"),
  saveRepoDatasetsBtn: document.getElementById("save-repo-datasets-btn"),
  repoSaveStatus: document.getElementById("repo-save-status"),

  trainBtn: document.getElementById("train-btn"),
  trainStatus: document.getElementById("train-status"),
  valAcc: document.getElementById("val-acc"),
  valLoss: document.getElementById("val-loss"),
  trainProgress: document.getElementById("train-progress"),
  trainNote: document.getElementById("train-note"),
  confusionCanvas: document.getElementById("confusion-canvas"),

  translateInputRow: document.getElementById("translate-input-row"),
  addSpaceBtn: document.getElementById("add-space-btn"),
  translateBtn: document.getElementById("translate-btn"),
  clearTranslateBtn: document.getElementById("clear-translate-btn"),
  translationOutput: document.getElementById("translation-output"),
  translationConfidence: document.getElementById("translation-confidence"),
  translationCount: document.getElementById("translation-count"),
  translationDetail: document.getElementById("translation-detail"),

  testCanvas: document.getElementById("test-canvas"),
  testPredictBtn: document.getElementById("test-predict-btn"),
  testClearBtn: document.getElementById("test-clear-btn"),
  testPred: document.getElementById("test-pred"),
  testConf: document.getElementById("test-conf"),
  testEntropy: document.getElementById("test-entropy"),
  testMargin: document.getElementById("test-margin"),

  normalizedCanvas: document.getElementById("normalized-canvas"),
  saliencyCanvas: document.getElementById("saliency-canvas"),
  posteriorBars: document.getElementById("posterior-bars"),
  bayesBars: document.getElementById("bayes-bars"),
  contribList: document.getElementById("contrib-list"),

  analysisPopover: document.getElementById("analysis-popover"),
  analysisTitle: document.getElementById("analysis-title"),
  analysisSubtitle: document.getElementById("analysis-subtitle"),
  analysisConfidence: document.getElementById("analysis-confidence"),
  analysisEntropy: document.getElementById("analysis-entropy"),
  analysisMargin: document.getElementById("analysis-margin"),
  analysisNormalizedCanvas: document.getElementById("analysis-normalized-canvas"),
  analysisSaliencyCanvas: document.getElementById("analysis-saliency-canvas"),
  analysisProbCanvas: document.getElementById("analysis-prob-canvas"),
  analysisTopBars: document.getElementById("analysis-top-bars"),
  analysisReason: document.getElementById("analysis-reason"),

  aboutBayesCanvas: document.getElementById("about-bayes-canvas"),
  aboutEntropyCanvas: document.getElementById("about-entropy-canvas"),
};

function assertDom() {
  const required = [
    ["language-name-input", el.languageNameInput],
    ["language-select", el.languageSelect],
    ["guide-canvas", el.guideCanvas],
    ["dataset-canvas", el.datasetCanvas],
    ["save-repo-datasets-btn", el.saveRepoDatasetsBtn],
    ["repo-save-status", el.repoSaveStatus],
    ["train-btn", el.trainBtn],
    ["translate-input-row", el.translateInputRow],
    ["test-canvas", el.testCanvas],
    ["normalized-canvas", el.normalizedCanvas],
    ["saliency-canvas", el.saliencyCanvas],
    ["analysis-popover", el.analysisPopover],
    ["analysis-reason", el.analysisReason],
    ["about-bayes-canvas", el.aboutBayesCanvas],
    ["about-entropy-canvas", el.aboutEntropyCanvas],
  ];
  for (let i = 0; i < required.length; i += 1) {
    if (!required[i][1]) {
      throw new Error(`Missing element #${required[i][0]}`);
    }
  }
}

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function randn() {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
  }
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < arr.length; i += 1) {
    if (arr[i] > arr[best]) best = i;
  }
  return best;
}

function topKIndices(arr, k) {
  const idx = Array.from({ length: arr.length }, (_, i) => i);
  idx.sort((a, b) => arr[b] - arr[a]);
  return idx.slice(0, Math.max(1, Math.min(k, idx.length)));
}

function softmax(logits, out, temperature = 1) {
  const invT = 1 / temperature;
  let m = logits[0];
  for (let i = 1; i < logits.length; i += 1) {
    if (logits[i] > m) m = logits[i];
  }

  let sum = 0;
  for (let i = 0; i < logits.length; i += 1) {
    const v = Math.exp((logits[i] - m) * invT);
    out[i] = v;
    sum += v;
  }
  for (let i = 0; i < logits.length; i += 1) {
    out[i] /= sum;
  }
}

function entropyBits(probs) {
  let h = 0;
  for (let i = 0; i < probs.length; i += 1) {
    const p = probs[i];
    if (p > 0) h -= p * Math.log2(p);
  }
  return h;
}

function xavierInit(size, fanIn) {
  const std = Math.sqrt(2 / fanIn);
  const out = new Float32Array(size);
  for (let i = 0; i < size; i += 1) out[i] = randn() * std;
  return out;
}

function setPill(node, text, kind) {
  node.textContent = text;
  node.className = `pill ${kind}`;
}

function setRepoSaveStatus(text, kind) {
  if (!el.repoSaveStatus) return;
  setPill(el.repoSaveStatus, text, kind);
}

function typesetMath(node = document.body, retries = 60) {
  if (window.MathJax && typeof window.MathJax.typesetPromise === "function") {
    window.MathJax.typesetPromise([node]).catch((err) => {
      console.warn("MathJax typeset failed:", err);
    });
    return;
  }
  if (retries <= 0) return;
  setTimeout(() => typesetMath(node, retries - 1), 140);
}

function emptySamplesMap() {
  const map = {};
  for (let i = 0; i < LETTERS.length; i += 1) {
    map[LETTERS[i]] = [];
  }
  return map;
}

function sanitizeLanguageName(value) {
  const x = String(value || "").replace(/\s+/g, " ").trim();
  return x.slice(0, 60);
}

function createLanguageProfile(name = "Unnamed Language", id = null) {
  const nowPart = Date.now().toString(36);
  const randPart = Math.random().toString(36).slice(2, 8);
  return {
    id: id || `lang-${nowPart}-${randPart}`,
    name: sanitizeLanguageName(name) || "Unnamed Language",
    samplesByLetter: emptySamplesMap(),
    model: null,
    metrics: { valAcc: null, valLoss: null, trainedAt: null },
  };
}

function activeProfile() {
  if (!state.activeLanguageId) return null;
  return state.languages[state.activeLanguageId] || null;
}

function syncCurrentProfile() {
  const profile = activeProfile();
  if (!profile) return;
  profile.name = sanitizeLanguageName(state.languageName) || "Unnamed Language";
  profile.samplesByLetter = state.samplesByLetter;
  profile.model = state.model;
  profile.metrics = state.metrics;
}

function loadProfileToState(profileId) {
  const profile = state.languages[profileId];
  if (!profile) return false;
  syncCurrentProfile();
  state.activeLanguageId = profileId;
  state.languageName = sanitizeLanguageName(profile.name) || "Unnamed Language";
  state.samplesByLetter = profile.samplesByLetter || emptySamplesMap();
  state.model = profile.model || null;
  state.metrics = profile.metrics || { valAcc: null, valLoss: null, trainedAt: null };
  state.activeLetterIndex = 0;
  return true;
}

function ensureLanguageInitialized() {
  const ids = Object.keys(state.languages || {});
  if (!ids.length) {
    const base = createLanguageProfile("Unnamed Language");
    state.languages = { [base.id]: base };
    state.activeLanguageId = base.id;
    state.languageName = base.name;
    state.samplesByLetter = base.samplesByLetter;
    state.model = base.model;
    state.metrics = base.metrics;
    return;
  }
  if (!state.activeLanguageId || !state.languages[state.activeLanguageId]) {
    state.activeLanguageId = ids[0];
  }
  loadProfileToState(state.activeLanguageId);
}

function refreshLanguageSelectUi() {
  const ids = Object.keys(state.languages);
  el.languageSelect.innerHTML = "";
  for (let i = 0; i < ids.length; i += 1) {
    const id = ids[i];
    const profile = state.languages[id];
    if (!profile) continue;
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = profile.name || "Unnamed Language";
    el.languageSelect.appendChild(opt);
  }
  if (state.activeLanguageId && ids.includes(state.activeLanguageId)) {
    el.languageSelect.value = state.activeLanguageId;
  }
}

function createNewLanguage() {
  const raw = window.prompt("Name the new alien language:", "New Alien Language");
  if (raw == null) return;
  const name = sanitizeLanguageName(raw) || "New Alien Language";
  const profile = createLanguageProfile(name);
  state.languages[profile.id] = profile;
  loadProfileToState(profile.id);
  drawGuideLetter();
  resetTranslator();
  clearPredictionDisplays();
  saveState();
  updateUi();
}

function deleteActiveLanguage() {
  const ids = Object.keys(state.languages);
  if (ids.length <= 1) {
    window.alert("At least one language profile must remain.");
    return;
  }
  const profile = activeProfile();
  if (!profile) return;
  const ok = window.confirm(`Delete language profile \"${profile.name}\" and its dataset/model?`);
  if (!ok) return;

  delete state.languages[profile.id];
  const nextIds = Object.keys(state.languages);
  const nextId = nextIds[0];
  loadProfileToState(nextId);
  drawGuideLetter();
  resetTranslator();
  clearPredictionDisplays();
  saveState();
  updateUi();
}

function encodeSample(vec) {
  const bytes = new Uint8Array(INPUT_DIM);
  for (let i = 0; i < INPUT_DIM; i += 1) {
    bytes[i] = Math.round(clamp(vec[i], 0, 1) * 255);
  }
  let bin = "";
  for (let i = 0; i < bytes.length; i += 1) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function decodeSample(encoded) {
  try {
    const bin = atob(encoded);
    if (bin.length !== INPUT_DIM) return null;
    const vec = new Float32Array(INPUT_DIM);
    for (let i = 0; i < INPUT_DIM; i += 1) vec[i] = bin.charCodeAt(i) / 255;
    return vec;
  } catch {
    return null;
  }
}

function sanitizeIncomingSample(x) {
  if (typeof x === "string") return decodeSample(x) ? x : null;
  if (Array.isArray(x) && x.length === INPUT_DIM) {
    const vec = new Float32Array(INPUT_DIM);
    for (let i = 0; i < INPUT_DIM; i += 1) {
      const v = Number(x[i]);
      if (!Number.isFinite(v)) return null;
      vec[i] = clamp(v, 0, 1);
    }
    return encodeSample(vec);
  }
  return null;
}

function uint8ToBase64(bytes) {
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const sub = bytes.subarray(i, i + chunk);
    bin += String.fromCharCode(...sub);
  }
  return btoa(bin);
}

function base64ToUint8(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i += 1) out[i] = bin.charCodeAt(i);
  return out;
}

function float32ToBase64(arr) {
  const bytes = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  return uint8ToBase64(bytes);
}

function base64ToFloat32(b64, expectedLen) {
  const bytes = base64ToUint8(b64);
  const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const out = new Float32Array(buf);
  if (expectedLen != null && out.length !== expectedLen) {
    throw new Error(`Float32 length mismatch: ${out.length} vs ${expectedLen}`);
  }
  return out;
}

function packFloat32(arr) {
  return { len: arr.length, b64: float32ToBase64(arr) };
}

function unpackFloat32(obj) {
  if (!obj || typeof obj.b64 !== "string" || !Number.isFinite(obj.len)) {
    throw new Error("Invalid packed float");
  }
  return base64ToFloat32(obj.b64, obj.len);
}

function serializeModel(model) {
  if (!model) return null;
  return {
    kind: "zoogle-cnn-v1",
    Wc: packFloat32(model.Wc),
    bc: packFloat32(model.bc),
    Wd: packFloat32(model.Wd),
    bd: packFloat32(model.bd),
  };
}

function deserializeModel(data) {
  if (!data) return null;
  return {
    Wc: unpackFloat32(data.Wc),
    bc: unpackFloat32(data.bc),
    Wd: unpackFloat32(data.Wd),
    bd: unpackFloat32(data.bd),
  };
}

function buildStatePayload() {
  syncCurrentProfile();
  const profiles = [];
  const ids = Object.keys(state.languages);
  for (let i = 0; i < ids.length; i += 1) {
    const profile = state.languages[ids[i]];
    if (!profile) continue;
    profiles.push({
      id: profile.id,
      name: sanitizeLanguageName(profile.name) || "Unnamed Language",
      samplesByLetter: profile.samplesByLetter,
      metrics: profile.metrics || { valAcc: null, valLoss: null, trainedAt: null },
      model: serializeModel(profile.model),
    });
  }

  return {
    version: 2,
    activeLanguageId: state.activeLanguageId,
    profiles,
    savedAt: Date.now(),
  };
}

function queueServerSave(immediate = false) {
  if (languageApiAvailability === "unavailable") return;
  serverSavePending = true;
  if (serverSaveTimer) {
    clearTimeout(serverSaveTimer);
    serverSaveTimer = null;
  }
  if (immediate) {
    void flushServerSave();
    return;
  }
  serverSaveTimer = setTimeout(() => {
    serverSaveTimer = null;
    void flushServerSave();
  }, SERVER_SAVE_DEBOUNCE_MS);
}

async function flushServerSave() {
  if (languageApiAvailability === "unavailable") return;
  if (serverSaveInFlight) return;
  if (!serverSavePending) return;

  setRepoSaveStatus("Saving language datasets to repo files...", "training");
  serverSavePending = false;
  serverSaveInFlight = true;
  try {
    const payload = buildStatePayload();
    const resp = await fetch(LANGUAGES_API_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (resp.status === 404 || resp.status === 405) {
      languageApiAvailability = "unavailable";
      setRepoSaveStatus("Server API unavailable. Run server.py to write dataset files.", "pending");
      return;
    }
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    languageApiAvailability = "available";
    const n = Array.isArray(payload.profiles) ? payload.profiles.length : 0;
    setRepoSaveStatus(`Saved ${n} language profile${n === 1 ? "" : "s"} to data/languages`, "ready");
  } catch (err) {
    console.warn("Server language save failed:", err);
    serverSavePending = true;
    setRepoSaveStatus("Save failed; retrying automatically...", "pending");
    if (languageApiAvailability === "unknown" && err instanceof TypeError) {
      languageApiAvailability = "unavailable";
      setRepoSaveStatus("Server API unavailable. Run server.py to write dataset files.", "pending");
    }
  } finally {
    serverSaveInFlight = false;
    if (serverSavePending && !serverSaveTimer && languageApiAvailability !== "unavailable") {
      serverSaveTimer = setTimeout(() => {
        serverSaveTimer = null;
        void flushServerSave();
      }, 900);
    }
  }
}

const LANGUAGES_STATIC_BASE = "data/languages";

async function loadLanguagesFromStaticFiles() {
  try {
    const indexResp = await fetch(`${LANGUAGES_STATIC_BASE}/index.json`, { cache: "no-store" });
    if (!indexResp.ok) return null;
    const index = await indexResp.json();
    const ids = Array.isArray(index.profileIds) ? index.profileIds : [];
    if (!ids.length) return null;

    const profiles = [];
    for (let i = 0; i < ids.length; i += 1) {
      const id = String(ids[i]).trim();
      if (!/^[A-Za-z0-9_-]{1,120}$/.test(id)) continue;
      const resp = await fetch(`${LANGUAGES_STATIC_BASE}/${encodeURIComponent(id)}.json`, { cache: "no-store" });
      if (!resp.ok) continue;
      const raw = await resp.json();
      if (!raw || typeof raw !== "object") continue;
      const samplesByLetter = {};
      for (let j = 0; j < LETTERS.length; j += 1) {
        const letter = LETTERS[j];
        const arr = Array.isArray(raw.samplesByLetter?.[letter]) ? raw.samplesByLetter[letter] : [];
        const valid = [];
        for (let k = 0; k < arr.length; k += 1) {
          const s = sanitizeIncomingSample(arr[k]);
          if (s) valid.push(s);
        }
        samplesByLetter[letter] = valid;
      }
      profiles.push({
        id: raw.id || id,
        name: String(raw.name || "Unnamed").slice(0, 120),
        samplesByLetter,
        metrics: raw.metrics && typeof raw.metrics === "object"
          ? { valAcc: raw.metrics.valAcc, valLoss: raw.metrics.valLoss, trainedAt: raw.metrics.trainedAt }
          : { valAcc: null, valLoss: null, trainedAt: null },
        model: raw.model ?? null,
      });
    }
    if (!profiles.length) return null;
    return {
      profiles,
      activeLanguageId: typeof index.activeLanguageId === "string" ? index.activeLanguageId : profiles[0].id,
      savedAt: Date.now(),
    };
  } catch (e) {
    return null;
  }
}

async function hydrateStateFromServer() {
  if (languageApiAvailability === "unavailable") return;
  try {
    const resp = await fetch(LANGUAGES_API_PATH, {
      method: "GET",
      cache: "no-store",
      headers: { Accept: "application/json" },
    });
    if (resp.status === 404 || resp.status === 405) {
      languageApiAvailability = "unavailable";
      const fallback = await loadLanguagesFromStaticFiles();
      if (fallback && fallback.profiles.length > 0) {
        const payload = {
          version: 2,
          activeLanguageId: fallback.activeLanguageId,
          profiles: fallback.profiles,
          savedAt: fallback.savedAt,
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
        loadState();
        ensureLanguageInitialized();
        drawGuideLetter();
        clearPredictionDisplays();
        resetTranslator();
        updateUi();
      }
      return;
    }
    if (!resp.ok) {
      const fallback = await loadLanguagesFromStaticFiles();
      if (fallback && fallback.profiles.length > 0) {
        const payload = {
          version: 2,
          activeLanguageId: fallback.activeLanguageId,
          profiles: fallback.profiles,
          savedAt: fallback.savedAt,
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
        loadState();
        ensureLanguageInitialized();
        drawGuideLetter();
        clearPredictionDisplays();
        resetTranslator();
        updateUi();
      }
      return;
    }
    const remote = await resp.json();
    if (!remote || !Array.isArray(remote.profiles)) {
      const fallback = await loadLanguagesFromStaticFiles();
      if (fallback && fallback.profiles.length > 0) {
        const payload = {
          version: 2,
          activeLanguageId: fallback.activeLanguageId,
          profiles: fallback.profiles,
          savedAt: fallback.savedAt,
        };
        localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
        loadState();
        ensureLanguageInitialized();
        drawGuideLetter();
        clearPredictionDisplays();
        resetTranslator();
        updateUi();
      }
      return;
    }
    languageApiAvailability = "available";

    let localSavedAt = 0;
    try {
      const rawLocal = localStorage.getItem(STORAGE_KEY);
      if (rawLocal) {
        const parsedLocal = JSON.parse(rawLocal);
        if (parsedLocal && Number.isFinite(parsedLocal.savedAt)) {
          localSavedAt = Number(parsedLocal.savedAt);
        }
      }
    } catch {
      localSavedAt = 0;
    }
    const remoteSavedAt = Number.isFinite(remote.savedAt) ? Number(remote.savedAt) : 0;
    if (localSavedAt > remoteSavedAt + 1000) {
      queueServerSave(true);
      return;
    }

    if (remote.profiles.length > 0) {
      const payload = {
        version: 2,
        activeLanguageId: typeof remote.activeLanguageId === "string" ? remote.activeLanguageId : null,
        profiles: remote.profiles,
        savedAt: Date.now(),
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
      loadState();
      ensureLanguageInitialized();
      drawGuideLetter();
      clearPredictionDisplays();
      resetTranslator();
      updateUi();
      return;
    }

    // Server has no profiles yet: push current in-memory/local copy.
    queueServerSave(true);
  } catch (err) {
    console.warn("Server language load failed:", err);
    if (languageApiAvailability === "unknown" && err instanceof TypeError) {
      languageApiAvailability = "unavailable";
    }
    const fallback = await loadLanguagesFromStaticFiles();
    if (fallback && fallback.profiles.length > 0) {
      const payload = {
        version: 2,
        activeLanguageId: fallback.activeLanguageId,
        profiles: fallback.profiles,
        savedAt: fallback.savedAt,
      };
      localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
      loadState();
      ensureLanguageInitialized();
      drawGuideLetter();
      clearPredictionDisplays();
      resetTranslator();
      updateUi();
    }
  }
}

async function saveAllLanguagesToRepoFiles() {
  const payload = buildStatePayload();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));

  if (languageApiAvailability === "unavailable") {
    setRepoSaveStatus("Server API unavailable. Run server.py to write dataset files.", "pending");
    window.alert("Run server.py first so the app can write language datasets into data/languages/.");
    return;
  }

  el.saveRepoDatasetsBtn.disabled = true;
  setRepoSaveStatus("Saving language datasets to repo files...", "training");
  try {
    const resp = await fetch(LANGUAGES_API_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (resp.status === 404 || resp.status === 405) {
      languageApiAvailability = "unavailable";
      throw new Error("Server API unavailable. Run server.py to write dataset files.");
    }
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    languageApiAvailability = "available";
    const n = Array.isArray(payload.profiles) ? payload.profiles.length : 0;
    setRepoSaveStatus(`Saved ${n} language profile${n === 1 ? "" : "s"} to data/languages`, "ready");
  } catch (err) {
    const message = err instanceof Error ? err.message : "Save failed";
    setRepoSaveStatus(message, "pending");
    console.warn("Manual language dataset save failed:", err);
  } finally {
    el.saveRepoDatasetsBtn.disabled = false;
  }
}

function saveState() {
  const payload = buildStatePayload();
  localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));
  queueServerSave(false);
}

function loadState() {
  state.languages = {};
  state.samplesByLetter = emptySamplesMap();
  const raw = localStorage.getItem(STORAGE_KEY);
  if (!raw) {
    ensureLanguageInitialized();
    return;
  }

  try {
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") {
      ensureLanguageInitialized();
      return;
    }
    if (Number(parsed.version) >= 2 && Array.isArray(parsed.profiles)) {
      for (let i = 0; i < parsed.profiles.length; i += 1) {
        const row = parsed.profiles[i];
        if (!row || typeof row !== "object") continue;
        const profile = createLanguageProfile(row.name, row.id);

        if (row.samplesByLetter && typeof row.samplesByLetter === "object") {
          for (let j = 0; j < LETTERS.length; j += 1) {
            const letter = LETTERS[j];
            const arr = Array.isArray(row.samplesByLetter[letter]) ? row.samplesByLetter[letter] : [];
            const valid = [];
            for (let k = 0; k < arr.length; k += 1) {
              const sample = sanitizeIncomingSample(arr[k]);
              if (sample) valid.push(sample);
            }
            profile.samplesByLetter[letter] = valid;
          }
        }

        if (row.metrics && typeof row.metrics === "object") {
          profile.metrics = {
            valAcc: Number.isFinite(row.metrics.valAcc) ? row.metrics.valAcc : null,
            valLoss: Number.isFinite(row.metrics.valLoss) ? row.metrics.valLoss : null,
            trainedAt: Number.isFinite(row.metrics.trainedAt) ? row.metrics.trainedAt : null,
          };
        }

        if (row.model) {
          try {
            profile.model = deserializeModel(row.model);
          } catch (err) {
            console.warn("Model restore failed:", err);
          }
        }
        state.languages[profile.id] = profile;
      }
      state.activeLanguageId = typeof parsed.activeLanguageId === "string" ? parsed.activeLanguageId : null;
    } else {
      // Backward compatibility with v1 single-language format.
      const legacy = createLanguageProfile(parsed.languageName || "Unnamed Language");
      if (parsed.samplesByLetter && typeof parsed.samplesByLetter === "object") {
        for (let i = 0; i < LETTERS.length; i += 1) {
          const letter = LETTERS[i];
          const arr = Array.isArray(parsed.samplesByLetter[letter]) ? parsed.samplesByLetter[letter] : [];
          const valid = [];
          for (let j = 0; j < arr.length; j += 1) {
            const sample = sanitizeIncomingSample(arr[j]);
            if (sample) valid.push(sample);
          }
          legacy.samplesByLetter[letter] = valid;
        }
      }
      if (parsed.metrics && typeof parsed.metrics === "object") {
        legacy.metrics = {
          valAcc: Number.isFinite(parsed.metrics.valAcc) ? parsed.metrics.valAcc : null,
          valLoss: Number.isFinite(parsed.metrics.valLoss) ? parsed.metrics.valLoss : null,
          trainedAt: Number.isFinite(parsed.metrics.trainedAt) ? parsed.metrics.trainedAt : null,
        };
      }
      if (parsed.model) {
        try {
          legacy.model = deserializeModel(parsed.model);
        } catch (err) {
          console.warn("Legacy model restore failed:", err);
        }
      }
      state.languages[legacy.id] = legacy;
      state.activeLanguageId = legacy.id;
    }
  } catch (err) {
    console.warn("State load failed:", err);
  }
  ensureLanguageInitialized();
}

function clearCanvas(ctx, w, h) {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);
}

function setupDrawingPad(canvas, lineWidth, onStrokeEnd = null) {
  const ctx = canvas.getContext("2d");
  clearCanvas(ctx, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#111";
  ctx.lineWidth = lineWidth;

  let drawing = false;
  let last = null;

  function toPoint(evt) {
    const rect = canvas.getBoundingClientRect();
    const x = ((evt.clientX - rect.left) / rect.width) * canvas.width;
    const y = ((evt.clientY - rect.top) / rect.height) * canvas.height;
    return { x, y };
  }

  function onPointerDown(evt) {
    evt.preventDefault();
    drawing = true;
    last = toPoint(evt);
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(last.x + 0.001, last.y + 0.001);
    ctx.stroke();
    canvas.setPointerCapture(evt.pointerId);
  }

  function onPointerMove(evt) {
    if (!drawing) return;
    evt.preventDefault();
    const p = toPoint(evt);
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    last = p;
  }

  function onPointerUp(evt) {
    if (!drawing) return;
    evt.preventDefault();
    drawing = false;
    last = null;
    if (onStrokeEnd) onStrokeEnd();
    if (canvas.hasPointerCapture(evt.pointerId)) {
      canvas.releasePointerCapture(evt.pointerId);
    }
  }

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("pointerleave", onPointerUp);

  return {
    canvas,
    ctx,
    clear() {
      clearCanvas(ctx, canvas.width, canvas.height);
    },
    hasInk() {
      return Boolean(getInkBounds(canvas, ctx));
    },
  };
}

function toInkVectorFromCanvas(ctx, w, h) {
  const data = ctx.getImageData(0, 0, w, h).data;
  const vec = new Float32Array(w * h);
  for (let i = 0; i < vec.length; i += 1) {
    const r = data[i * 4];
    vec[i] = clamp((255 - r) / 255, 0, 1);
  }
  return vec;
}

function getInkBounds(canvas, ctx, threshold = 18) {
  const w = canvas.width;
  const h = canvas.height;
  const data = ctx.getImageData(0, 0, w, h).data;

  let minX = w;
  let minY = h;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const idx = (y * w + x) * 4;
      const ink = 255 - data[idx];
      if (ink <= threshold) continue;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
  }

  if (maxX < minX || maxY < minY) return null;
  return { minX, minY, maxX, maxY };
}

function centerOfMassShift(vec) {
  let mass = 0;
  let mx = 0;
  let my = 0;
  for (let y = 0; y < IMAGE_SIZE; y += 1) {
    for (let x = 0; x < IMAGE_SIZE; x += 1) {
      const v = vec[y * IMAGE_SIZE + x];
      mass += v;
      mx += x * v;
      my += y * v;
    }
  }
  if (mass < 1e-6) return vec;

  const cx = mx / mass;
  const cy = my / mass;
  const tx = Math.round((IMAGE_SIZE - 1) * 0.5 - cx);
  const ty = Math.round((IMAGE_SIZE - 1) * 0.5 - cy);
  if (tx === 0 && ty === 0) return vec;

  const out = new Float32Array(INPUT_DIM);
  for (let y = 0; y < IMAGE_SIZE; y += 1) {
    for (let x = 0; x < IMAGE_SIZE; x += 1) {
      const sx = x - tx;
      const sy = y - ty;
      if (sx < 0 || sy < 0 || sx >= IMAGE_SIZE || sy >= IMAGE_SIZE) continue;
      out[y * IMAGE_SIZE + x] = vec[sy * IMAGE_SIZE + sx];
    }
  }
  return out;
}

function preprocessCanvas(canvas, ctx) {
  const bounds = getInkBounds(canvas, ctx, 18);
  if (!bounds) return null;

  const rawW = bounds.maxX - bounds.minX + 1;
  const rawH = bounds.maxY - bounds.minY + 1;
  const pad = Math.max(8, Math.round(0.14 * Math.max(rawW, rawH)));

  const sx = clamp(bounds.minX - pad, 0, canvas.width - 1);
  const sy = clamp(bounds.minY - pad, 0, canvas.height - 1);
  const sw = clamp(rawW + 2 * pad, 1, canvas.width - sx);
  const sh = clamp(rawH + 2 * pad, 1, canvas.height - sy);

  const tiny = document.createElement("canvas");
  tiny.width = IMAGE_SIZE;
  tiny.height = IMAGE_SIZE;
  const tctx = tiny.getContext("2d");
  clearCanvas(tctx, IMAGE_SIZE, IMAGE_SIZE);

  const inner = IMAGE_SIZE - 6;
  const scale = inner / Math.max(sw, sh);
  const dw = sw * scale;
  const dh = sh * scale;
  const dx = (IMAGE_SIZE - dw) / 2;
  const dy = (IMAGE_SIZE - dh) / 2;

  tctx.drawImage(canvas, sx, sy, sw, sh, dx, dy, dw, dh);
  const vec = toInkVectorFromCanvas(tctx, IMAGE_SIZE, IMAGE_SIZE);
  return centerOfMassShift(vec);
}

function renderVectorToCanvas(vec, canvas, heat = false) {
  const ctx = canvas.getContext("2d");
  const tiny = document.createElement("canvas");
  tiny.width = IMAGE_SIZE;
  tiny.height = IMAGE_SIZE;
  const tctx = tiny.getContext("2d");
  const img = tctx.createImageData(IMAGE_SIZE, IMAGE_SIZE);

  for (let i = 0; i < INPUT_DIM; i += 1) {
    const p = i * 4;
    const v = clamp(vec[i], 0, 1);
    if (!heat) {
      const g = Math.round(255 * (1 - v));
      img.data[p] = g;
      img.data[p + 1] = g;
      img.data[p + 2] = g;
      img.data[p + 3] = 255;
    } else {
      const h = (1 - v) * 0.66;
      const rgb = hslToRgb(h, 1, 0.52);
      img.data[p] = rgb[0];
      img.data[p + 1] = rgb[1];
      img.data[p + 2] = rgb[2];
      img.data[p + 3] = 255;
    }
  }

  tctx.putImageData(img, 0, 0);
  clearCanvas(ctx, canvas.width, canvas.height);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tiny, 0, 0, canvas.width, canvas.height);
}

function hslToRgb(h, s, l) {
  let r;
  let g;
  let b;

  if (s === 0) {
    r = l;
    g = l;
    b = l;
  } else {
    const hue2rgb = (p, q, t) => {
      let x = t;
      if (x < 0) x += 1;
      if (x > 1) x -= 1;
      if (x < 1 / 6) return p + (q - p) * 6 * x;
      if (x < 1 / 2) return q;
      if (x < 2 / 3) return p + (q - p) * (2 / 3 - x) * 6;
      return p;
    };

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function bilinearSample(vec, x, y) {
  if (x < 0 || y < 0 || x >= IMAGE_SIZE - 1 || y >= IMAGE_SIZE - 1) return 0;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const tx = x - x0;
  const ty = y - y0;

  const v00 = vec[y0 * IMAGE_SIZE + x0];
  const v10 = vec[y0 * IMAGE_SIZE + x1];
  const v01 = vec[y1 * IMAGE_SIZE + x0];
  const v11 = vec[y1 * IMAGE_SIZE + x1];

  const a = v00 * (1 - tx) + v10 * tx;
  const b = v01 * (1 - tx) + v11 * tx;
  return a * (1 - ty) + b * ty;
}

function dilate3x3(vec) {
  const out = new Float32Array(INPUT_DIM);
  for (let y = 0; y < IMAGE_SIZE; y += 1) {
    for (let x = 0; x < IMAGE_SIZE; x += 1) {
      let best = 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        const yy = y + dy;
        if (yy < 0 || yy >= IMAGE_SIZE) continue;
        for (let dx = -1; dx <= 1; dx += 1) {
          const xx = x + dx;
          if (xx < 0 || xx >= IMAGE_SIZE) continue;
          const v = vec[yy * IMAGE_SIZE + xx];
          if (v > best) best = v;
        }
      }
      out[y * IMAGE_SIZE + x] = best;
    }
  }
  return out;
}

function augmentSample(vec) {
  const out = new Float32Array(INPUT_DIM);
  const cx = (IMAGE_SIZE - 1) * 0.5;
  const cy = (IMAGE_SIZE - 1) * 0.5;

  const tx = randn() * 1.2;
  const ty = randn() * 1.2;
  const angle = randn() * 0.13;
  const scale = clamp(1 + randn() * 0.08, 0.86, 1.18);
  const cosA = Math.cos(angle);
  const sinA = Math.sin(angle);

  for (let y = 0; y < IMAGE_SIZE; y += 1) {
    for (let x = 0; x < IMAGE_SIZE; x += 1) {
      const xr = (x - cx - tx) / scale;
      const yr = (y - cy - ty) / scale;
      const xs = cosA * xr + sinA * yr + cx;
      const ys = -sinA * xr + cosA * yr + cy;
      out[y * IMAGE_SIZE + x] = bilinearSample(vec, xs, ys);
    }
  }

  let aug = out;
  if (Math.random() < 0.35) aug = dilate3x3(aug);
  for (let i = 0; i < INPUT_DIM; i += 1) {
    aug[i] = clamp(aug[i] + randn() * 0.014, 0, 1);
  }
  return aug;
}

function buildDataset() {
  const trainPairs = [];
  const valPairs = [];
  let readyLetters = 0;

  for (let i = 0; i < LETTERS.length; i += 1) {
    const letter = LETTERS[i];
    const encoded = state.samplesByLetter[letter] || [];
    if (encoded.length < MIN_SAMPLES_PER_LETTER) continue;

    const vecs = [];
    for (let j = 0; j < encoded.length; j += 1) {
      const vec = decodeSample(encoded[j]);
      if (vec) vecs.push(centerOfMassShift(vec));
    }
    if (vecs.length < MIN_SAMPLES_PER_LETTER) continue;

    readyLetters += 1;
    shuffle(vecs);

    const valCount = Math.max(1, Math.floor(vecs.length * VALIDATION_SPLIT));
    const split = vecs.length - valCount;

    for (let j = 0; j < vecs.length; j += 1) {
      const x = vecs[j];
      if (j < split) {
        trainPairs.push({ x, y: i });
        for (let a = 0; a < TRAIN_AUGMENTATIONS; a += 1) {
          trainPairs.push({ x: augmentSample(x), y: i });
        }
      } else {
        valPairs.push({ x, y: i });
      }
    }
  }

  if (readyLetters < MIN_READY_LETTERS_FOR_TRAIN || trainPairs.length < 2 || valPairs.length < 2) {
    return null;
  }

  shuffle(trainPairs);
  shuffle(valPairs);

  return {
    trainX: trainPairs.map((p) => p.x),
    trainY: Int32Array.from(trainPairs.map((p) => p.y)),
    valX: valPairs.map((p) => p.x),
    valY: Int32Array.from(valPairs.map((p) => p.y)),
    readyLetters,
  };
}

function createModel(classCount) {
  return {
    Wc: xavierInit(CONV_FILTERS * KERNEL_SIZE * KERNEL_SIZE, KERNEL_SIZE * KERNEL_SIZE),
    bc: new Float32Array(CONV_FILTERS),
    Wd: xavierInit(classCount * FLAT_DIM, FLAT_DIM),
    bd: new Float32Array(classCount),
  };
}

function forward(model, x, classCount, cache, temperature = 1) {
  const { zConv, aConv, flat, logits, probs } = cache;

  for (let f = 0; f < CONV_FILTERS; f += 1) {
    const wBase = f * KERNEL_SIZE * KERNEL_SIZE;
    for (let oy = 0; oy < CONV_OUT; oy += 1) {
      const iy = oy * STRIDE;
      for (let ox = 0; ox < CONV_OUT; ox += 1) {
        const ix = ox * STRIDE;
        let sum = model.bc[f];

        for (let ky = 0; ky < KERNEL_SIZE; ky += 1) {
          const yPos = iy + ky;
          for (let kx = 0; kx < KERNEL_SIZE; kx += 1) {
            const xPos = ix + kx;
            const wIdx = wBase + ky * KERNEL_SIZE + kx;
            const inIdx = yPos * IMAGE_SIZE + xPos;
            sum += model.Wc[wIdx] * x[inIdx];
          }
        }

        const outIdx = (f * CONV_OUT + oy) * CONV_OUT + ox;
        zConv[outIdx] = sum;
        const act = sum > 0 ? sum : 0;
        aConv[outIdx] = act;
        flat[outIdx] = act;
      }
    }
  }

  for (let c = 0; c < classCount; c += 1) {
    let sum = model.bd[c];
    const base = c * FLAT_DIM;
    for (let j = 0; j < FLAT_DIM; j += 1) sum += model.Wd[base + j] * flat[j];
    logits[c] = sum;
  }

  softmax(logits, probs, temperature);
}

function evaluate(model, X, Y, classCount) {
  const cache = {
    zConv: new Float32Array(FLAT_DIM),
    aConv: new Float32Array(FLAT_DIM),
    flat: new Float32Array(FLAT_DIM),
    logits: new Float32Array(classCount),
    probs: new Float32Array(classCount),
  };

  const confusion = Array.from({ length: classCount }, () => Array(classCount).fill(0));

  let correct = 0;
  let loss = 0;
  for (let i = 0; i < X.length; i += 1) {
    forward(model, X[i], classCount, cache);
    const pred = argmax(cache.probs);
    const y = Y[i];

    confusion[y][pred] += 1;
    if (pred === y) correct += 1;
    loss += -Math.log(Math.max(cache.probs[y], 1e-8));
  }

  return {
    accuracy: correct / X.length,
    loss: loss / X.length,
    confusion,
  };
}

function drawConfusionMatrix(confusion, labels) {
  const ctx = el.confusionCanvas.getContext("2d");
  const width = el.confusionCanvas.width;
  const height = el.confusionCanvas.height;
  clearCanvas(ctx, width, height);

  const n = labels.length;
  const margin = 86;
  const cell = (width - margin - 14) / n;

  ctx.fillStyle = "#0f172a";
  ctx.font = "12px 'Helvetica Neue', Helvetica, Arial, sans-serif";
  ctx.fillText("Predicted", width / 2 - 24, 16);
  ctx.save();
  ctx.translate(14, height / 2 + 26);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("True", 0, 0);
  ctx.restore();

  ctx.font = "9px ui-monospace, monospace";
  for (let i = 0; i < n; i += 1) {
    ctx.fillStyle = "#4d5f78";
    ctx.fillText(labels[i], margin - 18, margin + i * cell + cell * 0.62);
    ctx.fillText(labels[i], margin + i * cell + cell * 0.28, margin - 10);
  }

  for (let r = 0; r < n; r += 1) {
    const rowSum = confusion[r].reduce((a, b) => a + b, 0) || 1;
    for (let c = 0; c < n; c += 1) {
      const value = confusion[r][c] / rowSum;
      const x = margin + c * cell;
      const y = margin + r * cell;
      const red = Math.floor(245 - 155 * value);
      const green = Math.floor(248 - 165 * value);
      const blue = Math.floor(255 - 115 * value);
      ctx.fillStyle = `rgb(${red}, ${green}, ${blue})`;
      ctx.fillRect(x, y, cell - 1.2, cell - 1.2);
    }
  }
}

function percentile98(vec) {
  const arr = Array.from(vec);
  arr.sort((a, b) => a - b);
  const idx = Math.floor(0.98 * (arr.length - 1));
  return arr[idx] || 0;
}

function smooth3x3(vec) {
  const out = new Float32Array(INPUT_DIM);
  for (let y = 0; y < IMAGE_SIZE; y += 1) {
    for (let x = 0; x < IMAGE_SIZE; x += 1) {
      let sum = 0;
      let cnt = 0;
      for (let dy = -1; dy <= 1; dy += 1) {
        const yy = y + dy;
        if (yy < 0 || yy >= IMAGE_SIZE) continue;
        for (let dx = -1; dx <= 1; dx += 1) {
          const xx = x + dx;
          if (xx < 0 || xx >= IMAGE_SIZE) continue;
          sum += vec[yy * IMAGE_SIZE + xx];
          cnt += 1;
        }
      }
      out[y * IMAGE_SIZE + x] = sum / cnt;
    }
  }
  return out;
}

function computeSaliencyByOcclusion(inputVec, scoreFn) {
  const saliency = new Float32Array(INPUT_DIM);
  const scratch = new Float32Array(inputVec);

  const patchRadius = 1;
  const step = 2;
  const baseScore = scoreFn(inputVec);

  for (let y = 0; y < IMAGE_SIZE; y += step) {
    for (let x = 0; x < IMAGE_SIZE; x += step) {
      const idx = y * IMAGE_SIZE + x;
      if (inputVec[idx] < 0.0005) continue;

      scratch.set(inputVec);
      for (let dy = -patchRadius; dy <= patchRadius; dy += 1) {
        const yy = y + dy;
        if (yy < 0 || yy >= IMAGE_SIZE) continue;
        for (let dx = -patchRadius; dx <= patchRadius; dx += 1) {
          const xx = x + dx;
          if (xx < 0 || xx >= IMAGE_SIZE) continue;
          scratch[yy * IMAGE_SIZE + xx] = 0;
        }
      }

      const drop = Math.max(0, baseScore - scoreFn(scratch));
      for (let yy = y; yy < Math.min(IMAGE_SIZE, y + step); yy += 1) {
        for (let xx = x; xx < Math.min(IMAGE_SIZE, x + step); xx += 1) {
          const outIdx = yy * IMAGE_SIZE + xx;
          saliency[outIdx] = Math.max(saliency[outIdx], drop);
        }
      }
    }
  }

  let map = smooth3x3(saliency);
  for (let i = 0; i < INPUT_DIM; i += 1) map[i] *= Math.sqrt(inputVec[i]);

  let maxVal = 0;
  for (let i = 0; i < INPUT_DIM; i += 1) if (map[i] > maxVal) maxVal = map[i];
  if (maxVal < 1e-7) return Float32Array.from(inputVec);

  const scale = Math.max(1e-8, percentile98(map), maxVal * 0.6);
  for (let i = 0; i < INPUT_DIM; i += 1) {
    const v = clamp(map[i] / scale, 0, 1);
    map[i] = Math.pow(v, 0.72);
  }
  return map;
}

function createBars(container, labels) {
  container.innerHTML = "";
  const rows = [];
  for (let i = 0; i < labels.length; i += 1) {
    const row = document.createElement("div");
    row.className = "bar-row";

    const left = document.createElement("span");
    left.className = "mono";
    left.textContent = labels[i];

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    track.appendChild(fill);

    const right = document.createElement("span");
    right.className = "mono";
    right.textContent = "-";

    row.appendChild(left);
    row.appendChild(track);
    row.appendChild(right);
    container.appendChild(row);
    rows.push({ fill, right });
  }
  return rows;
}

function updateBars(rows, probs) {
  for (let i = 0; i < rows.length; i += 1) {
    const pct = 100 * probs[i];
    rows[i].fill.style.width = `${pct.toFixed(2)}%`;
    rows[i].right.textContent = `${pct.toFixed(1)}%`;
  }
}

function renderContributions(predClass, cache) {
  const contrib = Array(CONV_FILTERS).fill(0);
  const base = predClass * FLAT_DIM;
  for (let j = 0; j < FLAT_DIM; j += 1) {
    const f = Math.floor(j / (CONV_OUT * CONV_OUT));
    contrib[f] += state.model.Wd[base + j] * cache.aConv[j];
  }

  const items = contrib.map((c, f) => ({ f, c }));
  items.sort((a, b) => Math.abs(b.c) - Math.abs(a.c));

  el.contribList.innerHTML = "";
  let maxAbs = 1e-8;
  for (let i = 0; i < items.length; i += 1) {
    maxAbs = Math.max(maxAbs, Math.abs(items[i].c));
  }

  for (let i = 0; i < items.length; i += 1) {
    const it = items[i];
    const row = document.createElement("div");
    row.className = "contrib-row";

    const left = document.createElement("span");
    left.className = "mono";
    left.textContent = `f${it.f}`;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${(Math.abs(it.c) / maxAbs) * 100}%`;
    fill.style.background = it.c >= 0
      ? "linear-gradient(90deg, #19a7ce, #1f7a8c)"
      : "linear-gradient(90deg, #ff7b7b, #c1121f)";
    track.appendChild(fill);

    const right = document.createElement("span");
    right.className = "mono";
    right.textContent = it.c >= 0 ? `+${it.c.toFixed(2)}` : it.c.toFixed(2);

    row.appendChild(left);
    row.appendChild(track);
    row.appendChild(right);
    el.contribList.appendChild(row);
  }
}

function readyLetterCount() {
  let n = 0;
  for (let i = 0; i < LETTERS.length; i += 1) {
    if ((state.samplesByLetter[LETTERS[i]] || []).length >= MIN_SAMPLES_PER_LETTER) n += 1;
  }
  return n;
}

function totalSampleCount() {
  let n = 0;
  for (let i = 0; i < LETTERS.length; i += 1) {
    n += (state.samplesByLetter[LETTERS[i]] || []).length;
  }
  return n;
}

function activeLetter() {
  return LETTERS[state.activeLetterIndex];
}

function drawGuideLetter() {
  const ctx = el.guideCanvas.getContext("2d");
  clearCanvas(ctx, el.guideCanvas.width, el.guideCanvas.height);

  const w = el.guideCanvas.width;
  const h = el.guideCanvas.height;
  ctx.strokeStyle = "rgba(26, 41, 61, 0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h * 0.62);
  ctx.lineTo(w, h * 0.62);
  ctx.stroke();

  ctx.fillStyle = "rgba(26, 41, 61, 0.22)";
  ctx.font = "bold 176px 'Caveat', cursive";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(activeLetter(), w * 0.5, h * 0.55);

  ctx.font = "600 18px 'Helvetica Neue', Helvetica, Arial, sans-serif";
  ctx.fillStyle = "rgba(26, 41, 61, 0.7)";
  ctx.fillText(`Map Symbol -> ${activeLetter()}`, w * 0.5, h - 18);
}

function updateLetterGrid() {
  el.letterGrid.innerHTML = "";
  for (let i = 0; i < LETTERS.length; i += 1) {
    const letter = LETTERS[i];
    const count = (state.samplesByLetter[letter] || []).length;

    const cell = document.createElement("button");
    cell.type = "button";
    cell.className = "letter-cell";
    if (count === 0) cell.classList.add("empty");
    else if (count < MIN_SAMPLES_PER_LETTER) cell.classList.add("partial");
    else cell.classList.add("ready");
    if (i === state.activeLetterIndex) cell.classList.add("active");

    const top = document.createElement("div");
    top.className = "top";
    top.textContent = letter;

    const bottom = document.createElement("div");
    bottom.className = "count";
    bottom.textContent = `${count}`;

    cell.appendChild(top);
    cell.appendChild(bottom);
    cell.addEventListener("click", () => {
      state.activeLetterIndex = i;
      drawGuideLetter();
      updateUi();
    });

    el.letterGrid.appendChild(cell);
  }
}

function updateUi() {
  refreshLanguageSelectUi();
  const name = sanitizeLanguageName(state.languageName) || "Unnamed Language";
  const total = totalSampleCount();
  const ready = readyLetterCount();
  const active = activeLetter();
  const activeCount = (state.samplesByLetter[active] || []).length;

  el.languageNameInput.value = name === "Unnamed Language" ? "" : name;
  el.translateLanguageName.textContent = name;
  el.activeLetterChip.textContent = active;
  el.activeLetterCount.textContent = String(activeCount);
  el.totalSamples.textContent = String(total);
  el.readyLetters.textContent = `${ready} / ${LETTERS.length}`;

  el.statLanguage.textContent = name;
  el.statTotal.textContent = String(total);
  el.statReady.textContent = `${ready} / ${LETTERS.length}`;

  if (state.model) {
    setPill(el.statModel, "Trained", "ready");
    setPill(el.trainStatus, "Trained", "ready");
  } else if (state.isTraining) {
    setPill(el.statModel, "Training", "training");
    setPill(el.trainStatus, "Training", "training");
  } else {
    setPill(el.statModel, "Not Trained", "pending");
    setPill(el.trainStatus, "Not Trained", "pending");
  }

  if (state.metrics.valAcc != null) el.valAcc.textContent = `${(100 * state.metrics.valAcc).toFixed(1)}%`;
  else el.valAcc.textContent = "-";

  if (state.metrics.valLoss != null) el.valLoss.textContent = state.metrics.valLoss.toFixed(3);
  else el.valLoss.textContent = "-";

  if (ready >= MIN_READY_LETTERS_FOR_TRAIN) {
    setPill(el.setupStatus, `Ready to train: ${ready} letters have >= ${MIN_SAMPLES_PER_LETTER} samples`, "ready");
  } else {
    setPill(el.setupStatus, `Need ${MIN_READY_LETTERS_FOR_TRAIN}+ letters with ${MIN_SAMPLES_PER_LETTER} samples`, "pending");
  }

  el.trainBtn.disabled = state.isTraining;
  el.testPredictBtn.disabled = !state.model;
  el.translateBtn.disabled = !state.model;
  if (el.deleteLanguageBtn) el.deleteLanguageBtn.disabled = Object.keys(state.languages).length <= 1;

  updateLetterGrid();
  drawAboutVisuals();
}

function switchView(view) {
  state.activeView = view;
  if (view !== "translate") hideAnalysisPopover();
  for (let i = 0; i < el.navButtons.length; i += 1) {
    const btn = el.navButtons[i];
    btn.classList.toggle("active", btn.dataset.view === view);
  }

  for (const [name, node] of Object.entries(el.views)) {
    node.classList.toggle("active", name === view);
  }
}

function applyTheme(theme) {
  const normalized = theme === "neon" ? "extraterrestrial" : theme;
  const t = THEME_SEQUENCE.includes(normalized) ? normalized : "light";
  el.body.setAttribute("data-theme", t);
  localStorage.setItem(THEME_KEY, t);
  if (el.themeToggle) el.themeToggle.textContent = `Theme: ${THEME_LABEL[t]}`;
}

function initTheme() {
  const saved = localStorage.getItem(THEME_KEY);
  applyTheme(saved);
  el.themeToggle.addEventListener("click", () => {
    const current = el.body.getAttribute("data-theme") || "light";
    const idx = THEME_SEQUENCE.indexOf(current);
    const next = THEME_SEQUENCE[(idx + 1) % THEME_SEQUENCE.length];
    applyTheme(next);
  });
}

function handleSaveSample() {
  const vec = preprocessCanvas(state.setupPad.canvas, state.setupPad.ctx);
  if (!vec) {
    setPill(el.setupStatus, "Draw a symbol before saving.", "pending");
    return;
  }

  const encoded = encodeSample(vec);
  const letter = activeLetter();
  state.samplesByLetter[letter].push(encoded);
  state.setupPad.clear();

  saveState();
  drawGuideLetter();
  updateUi();
}

function handleResetActiveLetter() {
  const letter = activeLetter();
  const count = (state.samplesByLetter[letter] || []).length;
  if (!count) return;
  const ok = window.confirm(`Delete all ${count} samples for ${letter}?`);
  if (!ok) return;

  state.samplesByLetter[letter] = [];
  saveState();
  updateUi();
}

function handleResetDataset() {
  const profile = activeProfile();
  const label = profile ? profile.name : "this language";
  const ok = window.confirm(`Delete dataset and trained model for \"${label}\"?`);
  if (!ok) return;

  state.samplesByLetter = emptySamplesMap();
  state.model = null;
  state.metrics = { valAcc: null, valLoss: null, trainedAt: null };
  state.trainProgress.value = 0;
  state.setupPad.clear();
  state.testPad.clear();
  clearPredictionDisplays();
  drawGuideLetter();
  resetTranslator();
  saveState();
  updateUi();
}

function exportDataset() {
  const payload = {
    version: 1,
    languageName: sanitizeLanguageName(state.languageName),
    letters: LETTERS.slice(),
    samplesByLetter: state.samplesByLetter,
    exportedAt: Date.now(),
  };

  const blob = new Blob([JSON.stringify(payload)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `zoogle_dataset_${new Date(payload.exportedAt).toISOString().replace(/[:.]/g, "-")}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function importDatasetFile(file) {
  if (!file) return;
  try {
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== "object") throw new Error("Invalid payload");

    const incoming = parsed.samplesByLetter;
    if (!incoming || typeof incoming !== "object") throw new Error("Missing samplesByLetter");

    const out = emptySamplesMap();
    for (let i = 0; i < LETTERS.length; i += 1) {
      const letter = LETTERS[i];
      const arr = Array.isArray(incoming[letter]) ? incoming[letter] : [];
      const valid = [];
      for (let j = 0; j < arr.length; j += 1) {
        const sample = sanitizeIncomingSample(arr[j]);
        if (sample) valid.push(sample);
      }
      out[letter] = valid;
    }

    state.samplesByLetter = out;
    state.languageName = sanitizeLanguageName(parsed.languageName || state.languageName);

    state.model = null;
    state.metrics = { valAcc: null, valLoss: null, trainedAt: null };
    state.trainProgress.value = 0;

    drawGuideLetter();
    resetTranslator();
    clearPredictionDisplays();
    saveState();
    updateUi();
  } catch (err) {
    console.error(err);
    window.alert("Import failed. Please choose a valid dataset JSON file.");
  } finally {
    el.importDatasetInput.value = "";
  }
}

function buildPredictionCache(classCount) {
  return {
    zConv: new Float32Array(FLAT_DIM),
    aConv: new Float32Array(FLAT_DIM),
    flat: new Float32Array(FLAT_DIM),
    logits: new Float32Array(classCount),
    probs: new Float32Array(classCount),
  };
}

async function trainModel() {
  const dataset = buildDataset();
  if (!dataset) {
    setPill(el.trainStatus, `Need ${MIN_READY_LETTERS_FOR_TRAIN}+ letters with ${MIN_SAMPLES_PER_LETTER} samples`, "pending");
    return;
  }

  state.isTraining = true;
  updateUi();
  setPill(el.trainStatus, "Training", "training");
  el.trainProgress.value = 0;
  el.trainNote.textContent = `Training with ${dataset.readyLetters} ready letters...`;

  const classCount = LETTERS.length;
  const model = createModel(classCount);

  const trainX = dataset.trainX;
  const trainY = dataset.trainY;
  const valX = dataset.valX;
  const valY = dataset.valY;

  const idx = Array.from({ length: trainX.length }, (_, i) => i);

  const cache = {
    zConv: new Float32Array(FLAT_DIM),
    aConv: new Float32Array(FLAT_DIM),
    flat: new Float32Array(FLAT_DIM),
    logits: new Float32Array(classCount),
    probs: new Float32Array(classCount),
    dLogits: new Float32Array(classCount),
    dFlat: new Float32Array(FLAT_DIM),
    dZConv: new Float32Array(FLAT_DIM),
  };

  const gWc = new Float32Array(model.Wc.length);
  const gbc = new Float32Array(model.bc.length);
  const gWd = new Float32Array(model.Wd.length);
  const gbd = new Float32Array(model.bd.length);

  try {
    for (let epoch = 0; epoch < EPOCHS; epoch += 1) {
      shuffle(idx);

      for (let start = 0; start < trainX.length; start += BATCH_SIZE) {
        gWc.fill(0);
        gbc.fill(0);
        gWd.fill(0);
        gbd.fill(0);

        const end = Math.min(trainX.length, start + BATCH_SIZE);
        const batchCount = end - start;

        for (let p = start; p < end; p += 1) {
          const sampleIndex = idx[p];
          const x = trainX[sampleIndex];
          const y = trainY[sampleIndex];

          forward(model, x, classCount, cache);

          for (let c = 0; c < classCount; c += 1) {
            cache.dLogits[c] = cache.probs[c] - (c === y ? 1 : 0);
            gbd[c] += cache.dLogits[c];
            const denseBase = c * FLAT_DIM;
            for (let j = 0; j < FLAT_DIM; j += 1) {
              gWd[denseBase + j] += cache.dLogits[c] * cache.flat[j];
            }
          }

          for (let j = 0; j < FLAT_DIM; j += 1) {
            let sum = 0;
            for (let c = 0; c < classCount; c += 1) {
              sum += model.Wd[c * FLAT_DIM + j] * cache.dLogits[c];
            }
            cache.dFlat[j] = sum;
            cache.dZConv[j] = cache.zConv[j] > 0 ? sum : 0;
          }

          for (let f = 0; f < CONV_FILTERS; f += 1) {
            const wBase = f * KERNEL_SIZE * KERNEL_SIZE;
            for (let oy = 0; oy < CONV_OUT; oy += 1) {
              const iy = oy * STRIDE;
              for (let ox = 0; ox < CONV_OUT; ox += 1) {
                const ix = ox * STRIDE;
                const outIdx = (f * CONV_OUT + oy) * CONV_OUT + ox;
                const g = cache.dZConv[outIdx];
                if (g === 0) continue;
                gbc[f] += g;

                for (let ky = 0; ky < KERNEL_SIZE; ky += 1) {
                  const yPos = iy + ky;
                  for (let kx = 0; kx < KERNEL_SIZE; kx += 1) {
                    const xPos = ix + kx;
                    const wIdx = wBase + ky * KERNEL_SIZE + kx;
                    const inIdx = yPos * IMAGE_SIZE + xPos;
                    gWc[wIdx] += g * x[inIdx];
                  }
                }
              }
            }
          }
        }

        const step = LEARNING_RATE / batchCount;
        for (let i = 0; i < model.Wc.length; i += 1) model.Wc[i] -= step * gWc[i];
        for (let i = 0; i < model.bc.length; i += 1) model.bc[i] -= step * gbc[i];
        for (let i = 0; i < model.Wd.length; i += 1) model.Wd[i] -= step * gWd[i];
        for (let i = 0; i < model.bd.length; i += 1) model.bd[i] -= step * gbd[i];
      }

      const val = evaluate(model, valX, valY, classCount);
      state.metrics.valAcc = val.accuracy;
      state.metrics.valLoss = val.loss;
      el.valAcc.textContent = `${(100 * val.accuracy).toFixed(1)}%`;
      el.valLoss.textContent = val.loss.toFixed(3);
      drawConfusionMatrix(val.confusion, LETTERS);

      const pct = ((epoch + 1) / EPOCHS) * 100;
      el.trainProgress.value = pct;
      el.trainNote.textContent = `Epoch ${epoch + 1}/${EPOCHS} • Val Acc ${(100 * val.accuracy).toFixed(1)}%`;

      await new Promise((resolve) => setTimeout(resolve, 0));
    }

    state.model = model;
    state.metrics.trainedAt = Date.now();
    saveState();
    setPill(el.trainStatus, "Trained", "ready");
  } catch (err) {
    console.error(err);
    state.model = null;
    state.metrics.valAcc = null;
    state.metrics.valLoss = null;
    setPill(el.trainStatus, "Training Failed", "pending");
    window.alert("Training failed. See console for details.");
  } finally {
    state.isTraining = false;
    updateUi();
  }
}

function predictDistribution(vec, temp = INFERENCE_TEMPERATURE) {
  if (!state.model) return null;
  const classCount = LETTERS.length;
  const cache = buildPredictionCache(classCount);
  forward(state.model, vec, classCount, cache, temp);
  return cache;
}

function clearPredictionDisplays() {
  clearCanvas(el.normalizedCanvas.getContext("2d"), el.normalizedCanvas.width, el.normalizedCanvas.height);
  clearCanvas(el.saliencyCanvas.getContext("2d"), el.saliencyCanvas.width, el.saliencyCanvas.height);
  el.testPred.textContent = "-";
  el.testConf.textContent = "-";
  el.testEntropy.textContent = "-";
  el.testMargin.textContent = "-";
  el.contribList.innerHTML = "";

  const zeros = new Float32Array(LETTERS.length);
  updateBars(state.posteriorRows, zeros);
  updateBars(state.bayesRows, zeros);
}

function classifyTestLetter() {
  if (!state.model) {
    setPill(el.trainStatus, "Train the CNN first", "pending");
    return;
  }

  const vec = preprocessCanvas(state.testPad.canvas, state.testPad.ctx);
  if (!vec) return;

  renderVectorToCanvas(vec, el.normalizedCanvas, false);

  const classCount = LETTERS.length;
  const cache = buildPredictionCache(classCount);
  forward(state.model, vec, classCount, cache, INFERENCE_TEMPERATURE);

  const pred = argmax(cache.probs);
  const topIdx = topKIndices(cache.probs, 2);
  const conf = cache.probs[pred];
  const margin = cache.probs[topIdx[0]] - cache.probs[topIdx[1]];
  const ent = entropyBits(cache.probs);

  const tmp = buildPredictionCache(classCount);
  const saliency = computeSaliencyByOcclusion(vec, (probe) => {
    forward(state.model, probe, classCount, tmp, INFERENCE_TEMPERATURE);
    return tmp.probs[pred];
  });
  renderVectorToCanvas(saliency, el.saliencyCanvas, true);

  updateBars(state.posteriorRows, cache.probs);

  const prior = 1 / classCount;
  const evidence = new Float32Array(classCount);
  for (let i = 0; i < classCount; i += 1) {
    evidence[i] = clamp(cache.probs[i] / prior / 3, 0, 1);
  }
  updateBars(state.bayesRows, evidence);

  renderContributions(pred, cache);

  el.testPred.textContent = LETTERS[pred];
  el.testConf.textContent = `${(100 * conf).toFixed(1)}%`;
  el.testEntropy.textContent = `${ent.toFixed(2)} bits`;
  el.testMargin.textContent = `${(100 * margin).toFixed(1)}%`;
}

function createTranslateLetterToken() {
  const wrapper = document.createElement("div");
  wrapper.className = "translate-cell";

  const canvas = document.createElement("canvas");
  canvas.width = 78;
  canvas.height = 78;
  wrapper.appendChild(canvas);

  el.translateInputRow.appendChild(wrapper);

  const pad = setupDrawingPad(canvas, 7, () => {
    ensureTranslateTrailingBlank();
  });

  const token = {
    type: "letter",
    wrapper,
    canvas,
    ctx: pad.ctx,
    clear: pad.clear,
    hasInk: pad.hasInk,
  };

  canvas.addEventListener("pointerdown", () => {
    const last = state.translateTokens[state.translateTokens.length - 1];
    if (last === token) {
      setTimeout(() => ensureTranslateTrailingBlank(), 0);
    }
  });

  return token;
}

function createTranslateSpaceToken() {
  const node = document.createElement("div");
  node.className = "space-token";
  node.textContent = "SPACE";
  el.translateInputRow.appendChild(node);
  return {
    type: "space",
    node,
  };
}

function ensureTranslateTrailingBlank() {
  if (!state.translateTokens.length) {
    state.translateTokens.push(createTranslateLetterToken());
    return;
  }

  while (state.translateTokens.length >= 2) {
    const last = state.translateTokens[state.translateTokens.length - 1];
    const prev = state.translateTokens[state.translateTokens.length - 2];
    if (last.type === "letter" && prev.type === "letter" && !last.hasInk()) {
      last.wrapper.remove();
      state.translateTokens.pop();
    } else {
      break;
    }
  }

  const end = state.translateTokens[state.translateTokens.length - 1];
  if (end.type !== "letter" || end.hasInk()) {
    state.translateTokens.push(createTranslateLetterToken());
  }
}

function trimFinalBlankForTranslate() {
  while (state.translateTokens.length) {
    const last = state.translateTokens[state.translateTokens.length - 1];
    if (last.type === "letter" && !last.hasInk()) {
      last.wrapper.remove();
      state.translateTokens.pop();
      continue;
    }
    break;
  }
}

function resetTranslator() {
  el.translateInputRow.innerHTML = "";
  state.translateTokens = [];
  state.translateTokens.push(createTranslateLetterToken());
  state.lastTranslation = [];
  el.translationOutput.textContent = "-";
  el.translationConfidence.textContent = "-";
  el.translationCount.textContent = "0";
  el.translationDetail.innerHTML = "Train the model, then write symbols to decode with MAP rule \\(\\hat{y}=\\arg\\max_{y} p_{\\theta}(y\\mid X)\\).";
  typesetMath(el.translationDetail);
  hideAnalysisPopover();
}

function drawProbabilityChart(canvas, probs, predIdx) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  const n = probs.length;
  const padX = 10;
  const padY = 18;
  const barW = Math.max(2, (w - padX * 2) / n - 1);

  ctx.strokeStyle = "rgba(120,140,160,0.5)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(padX, h - padY + 0.5);
  ctx.lineTo(w - padX, h - padY + 0.5);
  ctx.stroke();

  for (let i = 0; i < n; i += 1) {
    const p = clamp(probs[i], 0, 1);
    const bh = p * (h - padY * 1.8);
    const x = padX + i * (barW + 1);
    const y = h - padY - bh;
    ctx.fillStyle = i === predIdx ? "#1f7a8c" : "#8bb9c8";
    ctx.fillRect(x, y, barW, bh);
  }

  ctx.fillStyle = "#23354a";
  ctx.font = "10px ui-monospace, monospace";
  for (let i = 0; i < n; i += 1) {
    if (i % 3 !== 0) continue;
    const x = padX + i * (barW + 1);
    ctx.fillText(LETTERS[i], x, h - 4);
  }
}

function aboutPosteriorSnapshot() {
  for (let i = 0; i < state.lastTranslation.length; i += 1) {
    const row = state.lastTranslation[i];
    if (!row || row.type !== "letter" || !row.probs) continue;
    const top = topKIndices(row.probs, 5);
    const labels = top.map((idx) => LETTERS[idx]);
    const posterior = top.map((idx) => clamp(row.probs[idx], 0, 1));
    const prior = Array(labels.length).fill(1 / LETTERS.length);
    return { labels, posterior, prior };
  }
  return {
    labels: ["A", "E", "O", "R", "N"],
    posterior: [0.56, 0.19, 0.11, 0.08, 0.06],
    prior: Array(5).fill(1 / LETTERS.length),
  };
}

function drawAboutBayesCanvas() {
  const canvas = el.aboutBayesCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  const snap = aboutPosteriorSnapshot();
  const n = snap.labels.length;
  const margin = { l: 46, r: 20, t: 24, b: 34 };
  const plotW = w - margin.l - margin.r;
  const plotH = h - margin.t - margin.b;
  const yMax = Math.max(0.15, ...snap.posterior) * 1.15;

  ctx.strokeStyle = "rgba(94,114,138,0.35)";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g += 1) {
    const y = margin.t + (plotH * g) / 4;
    ctx.beginPath();
    ctx.moveTo(margin.l, y);
    ctx.lineTo(w - margin.r, y);
    ctx.stroke();
    const val = yMax * (1 - g / 4);
    ctx.fillStyle = "#576a82";
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(val.toFixed(2), 8, y + 3);
  }

  const groupW = plotW / n;
  const barW = Math.min(24, groupW * 0.28);
  for (let i = 0; i < n; i += 1) {
    const xCenter = margin.l + groupW * (i + 0.5);
    const priorH = (snap.prior[i] / yMax) * plotH;
    const postH = (snap.posterior[i] / yMax) * plotH;

    ctx.fillStyle = "#b9c7d7";
    ctx.fillRect(xCenter - barW - 2, margin.t + plotH - priorH, barW, priorH);

    ctx.fillStyle = "#1f7a8c";
    ctx.fillRect(xCenter + 2, margin.t + plotH - postH, barW, postH);

    ctx.fillStyle = "#23354a";
    ctx.font = "11px ui-monospace, monospace";
    ctx.fillText(snap.labels[i], xCenter - 4, h - 12);
  }

  ctx.fillStyle = "#637a93";
  ctx.font = "11px 'Helvetica Neue', Helvetica, Arial, sans-serif";
  ctx.fillRect(w - 162, 10, 10, 10);
  ctx.fillText("Prior p(Y)", w - 146, 19);
  ctx.fillStyle = "#1f7a8c";
  ctx.fillRect(w - 86, 10, 10, 10);
  ctx.fillStyle = "#637a93";
  ctx.fillText("Posterior p(Y|X)", w - 70, 19);
}

function binaryEntropyBits(p) {
  const q = clamp(p, 1e-6, 1 - 1e-6);
  return -(q * Math.log2(q) + (1 - q) * Math.log2(1 - q));
}

function drawAboutEntropyCanvas() {
  const canvas = el.aboutEntropyCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, w, h);

  const margin = { l: 42, r: 18, t: 20, b: 34 };
  const plotW = w - margin.l - margin.r;
  const plotH = h - margin.t - margin.b;

  ctx.strokeStyle = "rgba(94,114,138,0.35)";
  ctx.lineWidth = 1;
  for (let g = 0; g <= 4; g += 1) {
    const y = margin.t + (plotH * g) / 4;
    ctx.beginPath();
    ctx.moveTo(margin.l, y);
    ctx.lineTo(w - margin.r, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#1f7a8c";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= 260; i += 1) {
    const p = i / 260;
    const x = margin.l + p * plotW;
    const y = margin.t + (1 - binaryEntropyBits(p)) * plotH;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const snap = aboutPosteriorSnapshot();
  const pStar = Math.max(...snap.posterior);
  const hStar = binaryEntropyBits(pStar);
  const xStar = margin.l + pStar * plotW;
  const yStar = margin.t + (1 - hStar) * plotH;

  ctx.fillStyle = "#d63b2f";
  ctx.beginPath();
  ctx.arc(xStar, yStar, 4, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#23354a";
  ctx.font = "11px 'Helvetica Neue', Helvetica, Arial, sans-serif";
  ctx.fillText(`p*=${pStar.toFixed(2)}, H2=${hStar.toFixed(2)} bits`, Math.min(w - 190, xStar + 8), Math.max(16, yStar - 8));

  ctx.fillStyle = "#576a82";
  ctx.font = "10px ui-monospace, monospace";
  ctx.fillText("0", margin.l - 4, h - 8);
  ctx.fillText("1", w - margin.r - 4, h - 8);
  ctx.fillText("0", 10, h - margin.b + 4);
  ctx.fillText("1", 10, margin.t + 4);
}

function drawAboutVisuals() {
  drawAboutBayesCanvas();
  drawAboutEntropyCanvas();
}

function renderTopProbRows(probs) {
  el.analysisTopBars.innerHTML = "";
  const top = topKIndices(probs, 6);
  for (let i = 0; i < top.length; i += 1) {
    const idx = top[i];
    const p = clamp(probs[idx], 0, 1);

    const row = document.createElement("div");
    row.className = "analysis-top-row";

    const left = document.createElement("span");
    left.className = "mono";
    left.textContent = LETTERS[idx];

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${(100 * p).toFixed(2)}%`;
    track.appendChild(fill);

    const right = document.createElement("span");
    right.className = "mono";
    right.textContent = `${(100 * p).toFixed(1)}%`;

    row.appendChild(left);
    row.appendChild(track);
    row.appendChild(right);
    el.analysisTopBars.appendChild(row);
  }
}

function qualitativeUncertainty(entropy, margin) {
  if (entropy <= 1.1 && margin >= 0.42) return "low";
  if (entropy <= 2.0 && margin >= 0.22) return "moderate";
  return "high";
}

function buildDecisionNarrative(entry) {
  const top = topKIndices(entry.probs, 2);
  const best = top[0];
  const next = top[1] ?? best;
  const pBest = clamp(entry.probs[best], 0, 1);
  const pNext = clamp(entry.probs[next], 0, 1);
  const uncertainty = qualitativeUncertainty(entry.entropy, entry.margin);
  return [
    `The CNN predicts \\(\\hat{y}=\\mathrm{${LETTERS[best]}}\\) because \\(p_{\\theta}(Y=\\mathrm{${LETTERS[best]}}\\mid X)=${pBest.toFixed(3)}\\).`,
    `The closest alternative is \\(\\mathrm{${LETTERS[next]}}\\) with \\(p_{\\theta}(Y=\\mathrm{${LETTERS[next]}}\\mid X)=${pNext.toFixed(3)}\\), giving margin \\(\\Delta p=${(100 * entry.margin).toFixed(1)}\\%\\).`,
    `Uncertainty is \\(H(Y\\mid X)=${entry.entropy.toFixed(2)}\\,\\text{bits}\\), so this decision is ${uncertainty}.`,
    "The saliency map highlights pixels where occlusion most reduced \\(p_{\\theta}(\\hat{y}\\mid X)\\)."
  ].join(" ");
}

function renderTranslationOutputEntries(entries) {
  el.translationOutput.innerHTML = "";
  if (!entries.length) {
    el.translationOutput.textContent = "-";
    return;
  }

  for (let i = 0; i < entries.length; i += 1) {
    const item = entries[i];
    if (item.type === "space") {
      el.translationOutput.appendChild(document.createTextNode(" "));
      continue;
    }
    const span = document.createElement("span");
    span.className = "output-letter";
    span.textContent = item.char;
    span.dataset.analysisIndex = String(i);
    span.title = `${item.char} • ${(100 * item.confidence).toFixed(1)}%`;
    el.translationOutput.appendChild(span);
  }
}

function hideAnalysisPopover() {
  if (state.popoverHideTimer) {
    clearTimeout(state.popoverHideTimer);
    state.popoverHideTimer = null;
  }
  state.popoverActiveIndex = -1;
  el.analysisPopover.classList.add("hidden");
}

function scheduleHideAnalysisPopover() {
  if (state.popoverHideTimer) clearTimeout(state.popoverHideTimer);
  state.popoverHideTimer = setTimeout(() => {
    state.popoverHideTimer = null;
    el.analysisPopover.classList.add("hidden");
  }, 120);
}

function positionAnalysisPopoverAtCursor(clientX, clientY) {
  const pop = el.analysisPopover;
  const pw = pop.offsetWidth || 420;
  const ph = pop.offsetHeight || 260;
  const gap = 14;

  let left = clientX + gap;
  let top = clientY + gap;
  if (left + pw > window.innerWidth - 8) left = clientX - pw - gap;
  if (top + ph > window.innerHeight - 8) top = clientY - ph - gap;
  left = clamp(left, 8, window.innerWidth - pw - 8);
  top = clamp(top, 8, window.innerHeight - ph - 8);
  pop.style.left = `${left}px`;
  pop.style.top = `${top}px`;
}

function showAnalysisPopover(index, clientX, clientY) {
  if (state.popoverHideTimer) {
    clearTimeout(state.popoverHideTimer);
    state.popoverHideTimer = null;
  }
  if (!state.model) return;
  const entry = state.lastTranslation[index];
  if (!entry || entry.type !== "letter") return;

  const isSameLetter = state.popoverActiveIndex === index;
  if (!isSameLetter) {
    if (!entry.saliency) {
      const classCount = LETTERS.length;
      const tmp = buildPredictionCache(classCount);
      const vec = entry.vec;
      entry.saliency = computeSaliencyByOcclusion(vec, (probe) => {
        forward(state.model, probe, classCount, tmp, INFERENCE_TEMPERATURE);
        return tmp.probs[entry.predIndex];
      });
    }

    renderVectorToCanvas(entry.vec, el.analysisNormalizedCanvas, false);
    renderVectorToCanvas(entry.saliency, el.analysisSaliencyCanvas, true);
    drawProbabilityChart(el.analysisProbCanvas, entry.probs, entry.predIndex);
    renderTopProbRows(entry.probs);

    el.analysisTitle.textContent = `MAP Decision: ${entry.char}`;
    el.analysisSubtitle.textContent = `Token #${entry.seqIndex + 1}`;
    el.analysisConfidence.textContent = `${(100 * entry.confidence).toFixed(1)}%`;
    el.analysisEntropy.textContent = `${entry.entropy.toFixed(2)} bits`;
    el.analysisMargin.textContent = `${(100 * entry.margin).toFixed(1)}%`;
    el.analysisReason.innerHTML = buildDecisionNarrative(entry);
    typesetMath(el.analysisPopover);
    state.popoverActiveIndex = index;
  }

  const pop = el.analysisPopover;
  pop.classList.remove("hidden");
  if (Number.isFinite(clientX) && Number.isFinite(clientY)) {
    positionAnalysisPopoverAtCursor(clientX, clientY);
  }
}

function addSpaceToken() {
  trimFinalBlankForTranslate();
  if (!state.translateTokens.length) {
    state.translateTokens.push(createTranslateLetterToken());
    return;
  }
  const last = state.translateTokens[state.translateTokens.length - 1];
  if (last.type === "space") return;

  state.translateTokens.push(createTranslateSpaceToken());
  state.translateTokens.push(createTranslateLetterToken());
}

function translateSequence() {
  if (!state.model) {
    setPill(el.trainStatus, "Train the CNN first", "pending");
    return;
  }

  trimFinalBlankForTranslate();
  if (!state.translateTokens.length) {
    resetTranslator();
    return;
  }

  const parts = [];
  const entries = [];
  const details = [];
  let sumConf = 0;
  let letterCount = 0;
  let sequenceCounter = 0;

  for (let i = 0; i < state.translateTokens.length; i += 1) {
    const token = state.translateTokens[i];
    if (token.type === "space") {
      if (parts.length && parts[parts.length - 1] !== " ") parts.push(" ");
      entries.push({ type: "space" });
      continue;
    }

    if (!token.hasInk()) continue;
    const vec = preprocessCanvas(token.canvas, token.ctx);
    if (!vec) continue;

    const cache = predictDistribution(vec, INFERENCE_TEMPERATURE);
    if (!cache) continue;

    const pred = argmax(cache.probs);
    const topIdx = topKIndices(cache.probs, 2);
    const ch = LETTERS[pred];
    const conf = cache.probs[pred];
    const ent = entropyBits(cache.probs);
    const margin = cache.probs[topIdx[0]] - cache.probs[topIdx[1]];

    parts.push(ch);
    details.push(`${ch} (${(100 * conf).toFixed(1)}%)`);
    entries.push({
      type: "letter",
      char: ch,
      confidence: conf,
      entropy: ent,
      margin,
      predIndex: pred,
      probs: Float32Array.from(cache.probs),
      vec: Float32Array.from(vec),
      saliency: null,
      seqIndex: sequenceCounter,
    });
    sequenceCounter += 1;
    sumConf += conf;
    letterCount += 1;
  }

  if (!letterCount) {
    state.lastTranslation = [];
    el.translationOutput.textContent = "-";
    el.translationConfidence.textContent = "-";
    el.translationCount.textContent = "0";
    el.translationDetail.textContent = "No valid symbols detected.";
    hideAnalysisPopover();
    ensureTranslateTrailingBlank();
    return;
  }

  const text = parts.join("").replace(/\s+/g, " ").trim();
  const avgConf = sumConf / letterCount;

  state.lastTranslation = entries;
  renderTranslationOutputEntries(entries);
  if (!text) el.translationOutput.textContent = "-";
  el.translationConfidence.textContent = `${(100 * avgConf).toFixed(1)}%`;
  el.translationCount.textContent = String(letterCount);
  el.translationDetail.textContent = details.join(" • ");
  hideAnalysisPopover();

  ensureTranslateTrailingBlank();
}

function bindEvents() {
  for (let i = 0; i < el.navButtons.length; i += 1) {
    const btn = el.navButtons[i];
    btn.addEventListener("click", () => switchView(btn.dataset.view));
  }

  el.languageNameInput.addEventListener("input", () => {
    state.languageName = sanitizeLanguageName(el.languageNameInput.value);
    saveState();
    updateUi();
  });

  el.languageSelect.addEventListener("change", () => {
    const targetId = el.languageSelect.value;
    if (!targetId || targetId === state.activeLanguageId) return;
    loadProfileToState(targetId);
    drawGuideLetter();
    resetTranslator();
    clearPredictionDisplays();
    saveState();
    updateUi();
  });

  el.newLanguageBtn.addEventListener("click", createNewLanguage);
  el.deleteLanguageBtn.addEventListener("click", deleteActiveLanguage);

  el.prevLetterBtn.addEventListener("click", () => {
    state.activeLetterIndex = (state.activeLetterIndex - 1 + LETTERS.length) % LETTERS.length;
    drawGuideLetter();
    updateUi();
  });

  el.nextLetterBtn.addEventListener("click", () => {
    state.activeLetterIndex = (state.activeLetterIndex + 1) % LETTERS.length;
    drawGuideLetter();
    updateUi();
  });

  el.saveSampleBtn.addEventListener("click", handleSaveSample);
  el.clearSampleBtn.addEventListener("click", () => state.setupPad.clear());

  el.resetLetterBtn.addEventListener("click", handleResetActiveLetter);
  el.resetDatasetBtn.addEventListener("click", handleResetDataset);

  el.exportDatasetBtn.addEventListener("click", exportDataset);
  el.importDatasetBtn.addEventListener("click", () => el.importDatasetInput.click());
  el.importDatasetInput.addEventListener("change", async (evt) => {
    const file = evt.target.files && evt.target.files[0];
    await importDatasetFile(file);
  });
  el.saveRepoDatasetsBtn.addEventListener("click", () => {
    void saveAllLanguagesToRepoFiles();
  });

  el.trainBtn.addEventListener("click", trainModel);

  el.translateBtn.addEventListener("click", translateSequence);
  el.addSpaceBtn.addEventListener("click", addSpaceToken);
  el.clearTranslateBtn.addEventListener("click", resetTranslator);

  el.translationOutput.addEventListener("mousemove", (evt) => {
    const target = evt.target.closest(".output-letter");
    if (!target || !el.translationOutput.contains(target)) {
      scheduleHideAnalysisPopover();
      return;
    }
    const idx = Number(target.dataset.analysisIndex);
    if (!Number.isInteger(idx) || idx < 0) {
      scheduleHideAnalysisPopover();
      return;
    }
    showAnalysisPopover(idx, evt.clientX, evt.clientY);
  });

  el.translationOutput.addEventListener("mouseleave", () => {
    hideAnalysisPopover();
  });

  el.testPredictBtn.addEventListener("click", classifyTestLetter);
  el.testClearBtn.addEventListener("click", () => {
    state.testPad.clear();
    clearPredictionDisplays();
  });

  window.addEventListener("resize", hideAnalysisPopover);
  window.addEventListener("scroll", hideAnalysisPopover, true);

  document.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter" && state.activeView === "translate") {
      evt.preventDefault();
      translateSequence();
    }
  });
}

function initialize() {
  assertDom();
  initTheme();
  loadState();
  ensureLanguageInitialized();
  if (!state.samplesByLetter) state.samplesByLetter = emptySamplesMap();

  state.setupPad = setupDrawingPad(el.datasetCanvas, 13, null);
  state.testPad = setupDrawingPad(el.testCanvas, 13, null);

  state.posteriorRows = createBars(el.posteriorBars, LETTERS);
  state.bayesRows = createBars(el.bayesBars, LETTERS);

  drawGuideLetter();
  clearPredictionDisplays();
  resetTranslator();

  bindEvents();
  updateUi();
  switchView("translate");
  typesetMath(document.body);
  void hydrateStateFromServer();
}

initialize();
