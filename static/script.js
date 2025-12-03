const video = document.getElementById('video');
const canvas = document.getElementById('capture');
const ctx = canvas.getContext('2d');
const predictionText = document.getElementById('prediction');
const audio = document.getElementById('tts-audio');

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
    })
    .catch(err => console.error("Webcam error:", err));

function captureAndPredict() {
    const width = video.videoWidth;
    const height = video.videoHeight;

    canvas.width = width;
    canvas.height = height;

    ctx.drawImage(video, 0, 0, width, height);
    const imageData = canvas.toDataURL('image/jpeg');

    // call the predict API request
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
    })
        .then(res => res.json())
        .then(data => {
            predictionText.innerText = `Prediction: ${data.prediction}`;

            // Show processed face sent to the model
            if (data.face) {
                const processedFace = document.getElementById('processed-face');
                processedFace.src = data.face;
                processedFace.style.display = 'block';
            }

            if (data.audio) {
                audio.src = data.audio;
                audio.load();
                audio.play();
            }
        })
        .catch(err => console.error("Prediction error:", err));
}
