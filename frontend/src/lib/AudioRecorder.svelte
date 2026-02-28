<script>
  let { onRecorded = () => {} } = $props();

  let isRecording = $state(false);
  let mediaRecorder = $state(null);
  let audioChunks = $state([]);
  let audioUrl = $state(null);
  let isUploading = $state(false);
  let error = $state(null);
  let recordingDuration = $state(0);
  let durationInterval = $state(null);

  async function startRecording() {
    error = null;
    audioUrl = null;
    audioChunks = [];

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunks.push(e.data);
        }
      };

      mediaRecorder.onstop = async () => {
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());

        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        audioUrl = URL.createObjectURL(blob);
        await uploadRecording(blob);
      };

      mediaRecorder.start();
      isRecording = true;
      recordingDuration = 0;
      durationInterval = setInterval(() => {
        recordingDuration += 1;
      }, 1000);
    } catch (e) {
      error = 'Microphone access denied. Please allow microphone access and try again.';
      console.error('Failed to start recording:', e);
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      isRecording = false;
      if (durationInterval) {
        clearInterval(durationInterval);
        durationInterval = null;
      }
    }
  }

  async function uploadRecording(blob) {
    isUploading = true;
    error = null;

    try {
      const formData = new FormData();
      formData.append('file', blob, 'recording.webm');

      const response = await fetch('/api/recording/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Upload failed');
      }

      const result = await response.json();
      onRecorded(result);
    } catch (e) {
      error = e.message;
    } finally {
      isUploading = false;
    }
  }

  function reRecord() {
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    audioUrl = null;
    audioChunks = [];
    error = null;
    recordingDuration = 0;
  }

  function formatDuration(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
</script>

<div class="audio-recorder">
  {#if !audioUrl}
    <div class="recorder-controls">
      {#if isRecording}
        <div class="recording-indicator">
          <span class="pulse-dot"></span>
          <span class="recording-time">{formatDuration(recordingDuration)}</span>
        </div>
        <button class="stop-btn" onclick={stopRecording}>
          Stop Recording
        </button>
      {:else}
        <button class="record-btn" onclick={startRecording}>
          Start Recording
        </button>
        <p class="hint">Click to start recording from your microphone</p>
      {/if}
    </div>
  {:else}
    <div class="preview">
      <audio controls src={audioUrl}></audio>
      {#if isUploading}
        <div class="uploading">
          <span class="spinner"></span>
          <span>Uploading...</span>
        </div>
      {:else}
        <button class="re-record-btn" onclick={reRecord}>
          Re-record
        </button>
      {/if}
    </div>
  {/if}

  {#if error}
    <p class="error">{error}</p>
  {/if}
</div>

<style>
  .audio-recorder {
    padding: 1rem;
  }

  .recorder-controls {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
    padding: 1.5rem;
  }

  .recording-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.25rem;
    font-weight: 500;
    color: #f44336;
  }

  .pulse-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: #f44336;
    animation: pulse 1s ease-in-out infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
  }

  .recording-time {
    font-family: monospace;
  }

  .record-btn {
    padding: 0.75rem 1.5rem;
    background: #f44336;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }

  .record-btn:hover {
    background: #d32f2f;
  }

  .stop-btn {
    padding: 0.75rem 1.5rem;
    background: #666;
    color: white;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s;
  }

  .stop-btn:hover {
    background: #555;
  }

  .hint {
    font-size: 0.875rem;
    color: #666;
    margin: 0;
  }

  .preview {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.75rem;
  }

  .preview audio {
    width: 100%;
    max-width: 400px;
  }

  .uploading {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #666;
  }

  .re-record-btn {
    padding: 0.5rem 1rem;
    background: #ff9800;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.875rem;
    transition: background 0.2s;
  }

  .re-record-btn:hover {
    background: #f57c00;
  }

  .error {
    color: #f44336;
    margin-top: 0.5rem;
    text-align: center;
    font-size: 0.875rem;
  }

  .spinner {
    display: inline-block;
    width: 16px;
    height: 16px;
    border: 2px solid #ccc;
    border-top-color: #4a90d9;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
