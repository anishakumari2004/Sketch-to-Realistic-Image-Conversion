// 1. Handle the Preview
document.getElementById("sketch").addEventListener("change", function(e) {
    const file = e.target.files[0];
    if (file) {
        document.getElementById("preview").src = URL.createObjectURL(file);
        document.getElementById("status").innerText = "Image uploaded. Ready to generate.";
    }
});

// 2. Handle the Generation
document.getElementById("genBtn").addEventListener("click", async function(event) {
    // STOP EVERYTHING - DO NOT REFRESH
    event.preventDefault();
    event.stopPropagation();

    const fileInput = document.getElementById("sketch");
    const promptInput = document.getElementById("prompt");
    const outputImg = document.getElementById("output");
    const statusText = document.getElementById("status");
    const btn = document.getElementById("genBtn");

    const file = fileInput.files[0];
    const prompt = promptInput.value;

    if (!file) {
        alert("Please upload a sketch first!");
        return;
    }

    // Lock UI
    btn.disabled = true;
    btn.innerText = "Processing...";
    statusText.innerText = "AI is working... Check your terminal progress bar.";
    console.log("Starting Fetch Request...");

    const formData = new FormData();
    formData.append("sketch", file);
    formData.append("prompt", prompt);

    try {
        const response = await fetch("http://127.0.0.1:5000/generate", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (data.image_url) {
            console.log("Success! Image URL:", data.image_url);
            outputImg.src = data.image_url + "?t=" + new Date().getTime();
            statusText.innerText = "Done! Image displayed below.";
        } else {
            statusText.innerText = "Error: " + data.error;
        }

    } catch (error) {
        console.error("Fetch Error:", error);
        statusText.innerText = "Connection error. Did the server restart?";
    } finally {
        btn.disabled = false;
        btn.innerText = "Generate";
    }
});