 let selectedTextType = "";  
 function selectType(type) {
  selectedTextType = type;
  currentOCRType = type === "Printed Text" ? "printed" : "handwritten";
  console.log("User selected OCR type:", selectedTextType);

  startContainer.style.display = "none";
  mainContainer.style.display = "block";
  subheading.textContent = `Upload an image with ${type}:`;
  currentStep = 1;
}




  const themeToggleButton = document.querySelector(".theme-toggle");
  const body = document.body;

  function toggleTheme() {
    body.classList.toggle("dark-mode");
    themeToggleButton.textContent = body.classList.contains("dark-mode") 
      ? "Switch to Light Mode" 
      : "Switch to Dark Mode";
  }

  const startContainer = document.getElementById("startContainer");
  const mainContainer = document.getElementById("mainContainer");
  const subheading = document.getElementById("subheading");
  const imageInput = document.getElementById("imageInput");
  const imagePreview = document.getElementById("imagePreview");
  const statusMessage = document.getElementById("statusMessage");
  const detectTextButton = document.getElementById("detectTextButton");
  const textSection = document.getElementById("textSection");
  const detectedText = document.getElementById("detectedText");
  const detectedLanguage = document.getElementById("detectedLanguage");

  let uploadedImage = "";
  let currentOCRType = "";
  let currentStep = 0;

  function goBack() {
    if (currentStep === 1) {
      mainContainer.style.display = "none";
      startContainer.style.display = "block";
      resetUploadSection();
      currentStep = 0;
    }
  }

  function resetUploadSection() {
    imageInput.value = "";
    imagePreview.innerHTML = "";
    statusMessage.textContent = "";
    detectTextButton.style.display = "none";
    textSection.style.display = "none";
    subheading.textContent = "";
  }

  imageInput.addEventListener("change", () => {
    const file = imageInput.files[0];
    if (file) {
      if (file.size > 2 * 1024 * 1024) {
        statusMessage.textContent = "❌ File too large. Upload under 2MB.";
        detectTextButton.style.display = "none";
        return;
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        uploadedImage = e.target.result;
        imagePreview.innerHTML = `<img src="${uploadedImage}" class="image-preview" alt="Uploaded Image">`;
        statusMessage.textContent = "✅ Image uploaded successfully!";
        detectTextButton.style.display = "block";
        textSection.style.display = "none";
      };
      reader.readAsDataURL(file);
    } else {
      statusMessage.textContent = "No file selected.";
      detectTextButton.style.display = "none";
    }
  });

  detectTextButton.addEventListener("click", () => {
  const file = imageInput.files[0];
  if (!file) {
    statusMessage.textContent = "Please upload an image first.";
    return;
  }

  const formData = new FormData();
  formData.append("image", file);
  formData.append("ocr_type", selectedTextType); 

  console.log("Sending OCR type:", selectedTextType); 
  statusMessage.textContent = "🔄 Processing image...";

  fetch(`{window.location.origin}/detect_text`, {
    method: "POST",
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    if (data.error) {
      statusMessage.textContent = data.warning;
      detectedText.innerText = "";
      detectedLanguage.innerText = "";
      textSection.style.display = "none";
      return;
    }

    if (data.placeholder) {
      statusMessage.textContent = "ℹ️ " + data.placeholder;
      textSection.style.display = "none";
      return;
    }

    if (!data.text) {
      statusMessage.textContent = "❌ No text found in the image.";
      textSection.style.display = "none";
      return;
    }

    detectedText.innerHTML = data.text.split("\n").join("<br>");
    detectedLanguage.innerText = "Language: " + (data.language || "Unknown");
    textSection.style.display = "block";
    detectTextButton.style.display = "none";
    statusMessage.textContent = "";
  })
  .catch(error => {
    console.error("Error:", error);
    statusMessage.textContent = "❌ Failed to process image.";
  });
});

  function copyText() {
    navigator.clipboard.writeText(detectedText.innerText).then(() => {
      alert("Text copied to clipboard!");
    });
  }

  window.addEventListener('beforeunload', () => {
  console.log("selectedTextType at unload:", selectedTextType);
});

