const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const previewBlock = document.getElementById("previewBlock");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const detectBtn = document.getElementById("detectBtn");
const confidenceRange = document.getElementById("confidenceRange");
const confidenceValue = document.getElementById("confidenceValue");
const errorBox = document.getElementById("errorBox");
const resultImageWrap = document.getElementById("resultImageWrap");
const resultImage = document.getElementById("resultImage");
const emptyState = document.getElementById("emptyState");
const detCount = document.getElementById("detCount");
const latency = document.getElementById("latency");
const imageSize = document.getElementById("imageSize");
const detectionsList = document.getElementById("detectionsList");
const healthStatus = document.getElementById("healthStatus");
const historyList = document.getElementById("historyList");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");
const statTotal = document.getElementById("statTotal");
const statLatency = document.getElementById("statLatency");
const statDetections = document.getElementById("statDetections");

let selectedFile = null;

function setError(message) {
  if (!message) {
    errorBox.classList.add("hidden");
    errorBox.textContent = "";
    return;
  }
  errorBox.classList.remove("hidden");
  errorBox.textContent = message;
}

function setSelectedFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    setError("Выберите файл изображения.");
    return;
  }

  selectedFile = file;
  setError("");
  previewBlock.classList.remove("hidden");
  previewImage.src = URL.createObjectURL(file);
  fileName.textContent = file.name;
  detectBtn.disabled = false;
}

dropzone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) {
    setSelectedFile(file);
  }
});

["dragenter", "dragover"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (file) {
    setSelectedFile(file);
  }
});

confidenceRange.addEventListener("input", () => {
  confidenceValue.textContent = Number(confidenceRange.value).toFixed(2);
});

async function checkHealth() {
  try {
    const response = await fetch("/health");
    const data = await response.json();
    healthStatus.textContent = `Сервис OK · ${data.architecture}`;
    healthStatus.classList.add("ok");
  } catch {
    healthStatus.textContent = "Сервис недоступен";
    healthStatus.classList.add("error");
  }
}

function renderDetections(detections) {
  detectionsList.innerHTML = "";
  if (!detections.length) {
    const item = document.createElement("li");
    item.textContent = "Объекты не найдены";
    detectionsList.appendChild(item);
    return;
  }

  detections.forEach((detection) => {
    const item = document.createElement("li");
    item.innerHTML = `
      <span class="class-name">${detection.class_name}</span>
      <span class="confidence">${(detection.confidence * 100).toFixed(1)}%</span>
    `;
    detectionsList.appendChild(item);
  });
}

async function runDetection() {
  if (!selectedFile) {
    return;
  }

  detectBtn.disabled = true;
  detectBtn.textContent = "Обработка...";
  setError("");

  const formData = new FormData();
  formData.append("file", selectedFile);

  try {
    const response = await fetch(`/predict?confidence=${confidenceRange.value}`, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Ошибка обработки");
    }

    emptyState.classList.add("hidden");
    resultImageWrap.classList.remove("hidden");
    resultImage.src = `data:image/jpeg;base64,${payload.annotated_image_base64}`;
    detCount.textContent = String(payload.detections_count);
    latency.textContent = `${payload.latency_ms.toFixed(1)} мс`;
    imageSize.textContent = `${payload.image_width}×${payload.image_height}`;
    renderDetections(payload.detections);
    await loadHistory();
    await loadStats();
  } catch (error) {
    setError(error.message);
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = "Найти объекты";
  }
}

async function loadStats() {
  const response = await fetch("/stats");
  const data = await response.json();
  statTotal.textContent = String(data.total_predictions);
  statLatency.textContent = `${data.average_latency_ms} мс`;
  statDetections.textContent = String(data.average_detections);
}

async function loadHistory() {
  const response = await fetch("/history?limit=10");
  const items = await response.json();
  historyList.innerHTML = "";

  if (!items.length) {
    historyList.innerHTML = '<div class="empty-state">История пока пуста.</div>';
    return;
  }

  items.forEach((item) => {
    const block = document.createElement("div");
    block.className = "history-item";
    block.innerHTML = `
      <div>
        <strong>${item.filename}</strong>
        <div class="meta">${new Date(item.created_at).toLocaleString("ru-RU")}</div>
      </div>
      <div class="meta">${item.detections_count} объектов · ${item.latency_ms} мс</div>
    `;
    historyList.appendChild(block);
  });
}

detectBtn.addEventListener("click", runDetection);
refreshHistoryBtn.addEventListener("click", async () => {
  await loadHistory();
  await loadStats();
});

checkHealth();
loadHistory();
loadStats();