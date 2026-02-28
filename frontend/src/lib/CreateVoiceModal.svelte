<script>
  import AudioRecorder from './AudioRecorder.svelte';

  let { onClose = () => {}, onCreated = () => {} } = $props();

  // Input mode tabs
  let inputMode = $state('search'); // 'search' | 'upload' | 'record'

  // Search mode state
  let searchQuery = $state('');
  let searchResults = $state([]);
  let isSearching = $state(false);
  let isDownloading = $state(false);
  let downloadingId = $state(null);

  // Upload mode state
  let isUploadingFile = $state(false);

  // Shared audio state
  let audioFilename = $state(null);
  let selectedVideoUrl = $state(null);
  let step = $state('input'); // 'input' | 'transcript'

  // Transcript editing state
  let transcript = $state('');
  let isTranscribing = $state(false);
  let voiceName = $state('');
  let language = $state('en');
  let selectedModel = $state('');
  let models = $state([]);
  let isCreating = $state(false);
  let error = $state(null);

  const languages = [
    { code: 'en', label: 'English' },
    { code: 'zh', label: 'Chinese' },
    { code: 'ja', label: 'Japanese' },
    { code: 'ko', label: 'Korean' },
    { code: 'de', label: 'German' },
    { code: 'fr', label: 'French' },
    { code: 'ru', label: 'Russian' },
    { code: 'pt', label: 'Portuguese' },
    { code: 'es', label: 'Spanish' },
    { code: 'it', label: 'Italian' }
  ];

  // Fetch models on mount
  $effect(() => {
    fetchModels();
  });

  async function fetchModels() {
    try {
      const response = await fetch('/api/cloned-voices/models');
      if (response.ok) {
        models = await response.json();
        if (models.length > 0) {
          selectedModel = models[0].id;
        }
      }
    } catch (e) {
      console.error('Failed to fetch models:', e);
    }
  }

  // YouTube search
  async function searchYoutube() {
    if (!searchQuery.trim()) return;

    isSearching = true;
    error = null;

    try {
      const response = await fetch(`/api/youtube/search?q=${encodeURIComponent(searchQuery)}`);
      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Search failed');
      }
      searchResults = await response.json();
    } catch (e) {
      error = e.message;
    } finally {
      isSearching = false;
    }
  }

  function handleSearchKeydown(e) {
    if (e.key === 'Enter') {
      searchYoutube();
    }
  }

  async function downloadYoutube(result) {
    isDownloading = true;
    downloadingId = result.id;
    error = null;

    try {
      const response = await fetch('/api/youtube/download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: result.url })
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Download failed');
      }

      const data = await response.json();
      audioFilename = data.filename;
      selectedVideoUrl = result.url;
      await autoTranscribe(data.filename);
    } catch (e) {
      error = e.message;
    } finally {
      isDownloading = false;
      downloadingId = null;
    }
  }

  // File upload
  async function handleFileUpload(e) {
    const files = e.target?.files;
    if (!files?.length) return;

    isUploadingFile = true;
    error = null;

    try {
      const formData = new FormData();
      formData.append('file', files[0]);

      const response = await fetch('/api/recording/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Upload failed');
      }

      const result = await response.json();
      audioFilename = result.filename;
      await autoTranscribe(result.filename);
    } catch (e) {
      error = e.message;
    } finally {
      isUploadingFile = false;
    }
  }

  // Recording callback
  async function handleRecorded(result) {
    audioFilename = result.filename;
    await autoTranscribe(result.filename);
  }

  // Auto-transcribe
  async function autoTranscribe(filename) {
    isTranscribing = true;
    error = null;

    try {
      const response = await fetch('/api/transcribe', {
        method: 'POST'
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Transcription failed');
      }

      const data = await response.json();
      transcript = data.text || '';
      step = 'transcript';
    } catch (e) {
      error = e.message;
    } finally {
      isTranscribing = false;
    }
  }

  // Create voice
  async function createVoice() {
    if (!voiceName.trim()) {
      error = 'Please enter a voice name';
      return;
    }
    if (!transcript.trim()) {
      error = 'Transcript cannot be empty';
      return;
    }

    isCreating = true;
    error = null;

    try {
      const sourceType = inputMode === 'search' ? 'youtube' : inputMode === 'upload' ? 'upload' : 'recording';
      const payload = {
          name: voiceName,
          audio_filename: audioFilename,
          transcript: transcript,
          language: language,
          model_id: selectedModel,
          source_type: sourceType
      };
      if (sourceType === 'youtube' && selectedVideoUrl) {
          payload.source_url = selectedVideoUrl;
      }

      const response = await fetch('/api/cloned-voices', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to create voice');
      }

      onCreated();
    } catch (e) {
      error = e.message;
    } finally {
      isCreating = false;
    }
  }

  function goBack() {
    step = 'input';
    transcript = '';
    audioFilename = null;
    selectedVideoUrl = null;
    error = null;
  }

  function handleOverlayClick(e) {
    if (e.target === e.currentTarget) {
      onClose();
    }
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="modal-overlay" onclick={handleOverlayClick}>
  <div class="modal-dialog">
    <div class="modal-header">
      <h2>{step === 'input' ? 'Create New Voice' : 'Edit Transcript'}</h2>
      <button class="close-btn" onclick={onClose}>&times;</button>
    </div>

    {#if step === 'input'}
      <div class="input-tabs">
        <button
          class="input-tab"
          class:active={inputMode === 'search'}
          onclick={() => { inputMode = 'search'; error = null; }}
        >
          Search by Name
        </button>
        <button
          class="input-tab"
          class:active={inputMode === 'upload'}
          onclick={() => { inputMode = 'upload'; error = null; }}
        >
          Upload File
        </button>
        <button
          class="input-tab"
          class:active={inputMode === 'record'}
          onclick={() => { inputMode = 'record'; error = null; }}
        >
          Record Voice
        </button>
      </div>

      <div class="modal-body">
        {#if inputMode === 'search'}
          <div class="search-section">
            <div class="search-bar">
              <input
                type="text"
                placeholder="Search YouTube for voice samples..."
                bind:value={searchQuery}
                onkeydown={handleSearchKeydown}
              />
              <button
                class="search-btn"
                onclick={searchYoutube}
                disabled={isSearching || !searchQuery.trim()}
              >
                {isSearching ? 'Searching...' : 'Search'}
              </button>
            </div>

            {#if searchResults.length > 0}
              <div class="search-results">
                {#each searchResults as result}
                  <div class="search-result">
                    <div class="result-info">
                      <span class="result-title">{result.title}</span>
                      <span class="result-duration">{result.duration}</span>
                    </div>
                    <button
                      class="select-btn"
                      onclick={() => downloadYoutube(result)}
                      disabled={isDownloading}
                    >
                      {downloadingId === result.id ? 'Downloading...' : 'Select'}
                    </button>
                  </div>
                {/each}
              </div>
            {/if}
          </div>

        {:else if inputMode === 'upload'}
          <div class="upload-section">
            {#if isUploadingFile}
              <div class="uploading-state">
                <span class="spinner"></span>
                <span>Uploading...</span>
              </div>
            {:else}
              <label class="file-upload-label">
                Choose Audio File
                <input type="file" accept="audio/*" onchange={handleFileUpload} />
              </label>
              <p class="hint">Supports MP3, WAV, M4A, OGG, FLAC, and other audio formats</p>
            {/if}
          </div>

        {:else if inputMode === 'record'}
          <AudioRecorder onRecorded={handleRecorded} />
        {/if}

        {#if isTranscribing}
          <div class="transcribing">
            <span class="spinner"></span>
            <span>Transcribing audio...</span>
          </div>
        {/if}
      </div>

    {:else if step === 'transcript'}
      <div class="modal-body">
        <div class="transcript-form">
          <div class="form-field">
            <label for="voice-name">Voice Name</label>
            <input
              type="text"
              id="voice-name"
              placeholder="Enter a name for this voice"
              bind:value={voiceName}
            />
          </div>

          <div class="form-field">
            <label for="transcript">Transcript</label>
            <textarea
              id="transcript"
              rows="6"
              placeholder="Edit the transcript if needed..."
              bind:value={transcript}
            ></textarea>
          </div>

          <div class="form-row">
            <div class="form-field">
              <label for="language">Language</label>
              <select id="language" bind:value={language}>
                {#each languages as lang}
                  <option value={lang.code}>{lang.label}</option>
                {/each}
              </select>
            </div>

            <div class="form-field">
              <label for="model">Model</label>
              <select id="model" bind:value={selectedModel}>
                {#each models as model}
                  <option value={model.id}>{model.name} ({model.vram_gb} GB)</option>
                {/each}
              </select>
            </div>
          </div>

          <div class="form-actions">
            <button class="back-btn" onclick={goBack}>Back</button>
            <button
              class="create-btn"
              onclick={createVoice}
              disabled={isCreating || !voiceName.trim() || !transcript.trim()}
            >
              {isCreating ? 'Creating...' : 'Create Voice'}
            </button>
          </div>
        </div>
      </div>
    {/if}

    {#if error}
      <div class="modal-error">
        <p>{error}</p>
      </div>
    {/if}
  </div>
</div>

<style>
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }

  .modal-dialog {
    background: white;
    border-radius: 12px;
    width: 90%;
    max-width: 600px;
    max-height: 85vh;
    overflow-y: auto;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  }

  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #e0e0e0;
  }

  .modal-header h2 {
    margin: 0;
    font-size: 1.25rem;
    color: #333;
  }

  .close-btn {
    background: none;
    border: none;
    font-size: 1.5rem;
    cursor: pointer;
    color: #666;
    padding: 0;
    line-height: 1;
  }

  .close-btn:hover {
    color: #333;
  }

  .input-tabs {
    display: flex;
    border-bottom: 1px solid #e0e0e0;
  }

  .input-tab {
    flex: 1;
    padding: 0.75rem;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 500;
    color: #666;
    transition: all 0.2s;
  }

  .input-tab:hover {
    color: #333;
    background: #f5f5f5;
  }

  .input-tab.active {
    color: #4a90d9;
    border-bottom-color: #4a90d9;
  }

  .modal-body {
    padding: 1.5rem;
  }

  /* Search mode */
  .search-bar {
    display: flex;
    gap: 0.5rem;
  }

  .search-bar input {
    flex: 1;
    padding: 0.5rem 0.75rem;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 0.875rem;
  }

  .search-bar input:focus {
    outline: none;
    border-color: #4a90d9;
    box-shadow: 0 0 0 2px rgba(74, 144, 217, 0.2);
  }

  .search-btn {
    padding: 0.5rem 1rem;
    background: #4a90d9;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.875rem;
    white-space: nowrap;
  }

  .search-btn:hover:not(:disabled) {
    background: #3a7bc8;
  }

  .search-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
  }

  .search-results {
    margin-top: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    max-height: 300px;
    overflow-y: auto;
  }

  .search-result {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    gap: 0.75rem;
  }

  .result-info {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    flex: 1;
    min-width: 0;
  }

  .result-title {
    font-weight: 500;
    font-size: 0.875rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .result-duration {
    font-size: 0.75rem;
    color: #666;
  }

  .select-btn {
    padding: 0.375rem 0.75rem;
    background: #4caf50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
    white-space: nowrap;
    flex-shrink: 0;
  }

  .select-btn:hover:not(:disabled) {
    background: #43a047;
  }

  .select-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
  }

  /* Upload mode */
  .upload-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 2rem;
    gap: 0.75rem;
  }

  .file-upload-label {
    display: inline-block;
    padding: 0.75rem 1.5rem;
    background: #4a90d9;
    color: white;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1rem;
    font-weight: 500;
    transition: background 0.2s;
  }

  .file-upload-label:hover {
    background: #3a7bc8;
  }

  .file-upload-label input {
    display: none;
  }

  .hint {
    font-size: 0.8rem;
    color: #888;
    margin: 0;
  }

  .uploading-state {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #666;
    font-size: 0.875rem;
  }

  .transcribing {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 1.5rem;
    padding: 1rem;
    background: #f5f5f5;
    border-radius: 8px;
    color: #666;
  }

  /* Transcript form */
  .transcript-form {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .form-field {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
  }

  .form-field label {
    font-size: 0.875rem;
    font-weight: 500;
    color: #333;
  }

  .form-field input,
  .form-field select {
    padding: 0.5rem 0.75rem;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 0.875rem;
  }

  .form-field input:focus,
  .form-field select:focus,
  .form-field textarea:focus {
    outline: none;
    border-color: #4a90d9;
    box-shadow: 0 0 0 2px rgba(74, 144, 217, 0.2);
  }

  .form-field textarea {
    padding: 0.5rem 0.75rem;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 0.875rem;
    font-family: inherit;
    resize: vertical;
  }

  .form-row {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }

  .form-actions {
    display: flex;
    justify-content: space-between;
    margin-top: 0.5rem;
  }

  .back-btn {
    padding: 0.5rem 1rem;
    background: #e0e0e0;
    color: #333;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.875rem;
    transition: background 0.2s;
  }

  .back-btn:hover {
    background: #d0d0d0;
  }

  .create-btn {
    padding: 0.5rem 1.5rem;
    background: #4caf50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 500;
    transition: background 0.2s;
  }

  .create-btn:hover:not(:disabled) {
    background: #43a047;
  }

  .create-btn:disabled {
    background: #ccc;
    cursor: not-allowed;
  }

  .modal-error {
    padding: 0 1.5rem 1rem;
  }

  .modal-error p {
    margin: 0;
    padding: 0.5rem 0.75rem;
    background: #ffebee;
    color: #f44336;
    border-radius: 4px;
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
