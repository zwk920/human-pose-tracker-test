const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const poseNet = posenet.load();
const handpose = handpose.load();
const facemesh = facemesh.load();

// Call the beginVideo() function to start the video stream
beginVideo();

async function beginVideo() {
  try {
    const videoConstraints = {
      facingMode: 'user',
      width: { ideal: 640 },
      height: { ideal: 480 },
    };
    const stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
    video.srcObject = stream;
    video.onloadedmetadata = () => {
      video.play();
      poseDetectionFrame();
    };
  } catch (err) {
    console.error('Error starting video:', err);
  }
}

async function poseDetectionFrame() {
  const pose = await poseNet.estimatePoses(video, {
    flipHorizontal: false,
    decodingMethod: "single-person",
  });

  const hand = await handpose.estimateHands(video, {
    flipHorizontal: false,
  });

  const face = await facemesh.estimateFaces(video, {
    flipHorizontal: false,
  });

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();
  ctx.scale(-1, 1);
  ctx.translate(-canvas.width, 0);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  ctx.restore();

  drawResults(pose, hand, face);

  requestAnimationFrame(poseDetectionFrame);
}

function drawResults(pose, hand, face) {
  // Draw pose, hand, and face results here
}

  