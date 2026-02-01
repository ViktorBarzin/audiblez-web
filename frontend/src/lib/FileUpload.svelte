<script>
  let { onUpload = () => {} } = $props();

  let isDragOver = $state(false);
  let uploadedFile = $state(null);
  let isUploading = $state(false);
  let error = $state(null);

  function handleDragOver(e) {
    e.preventDefault();
    isDragOver = true;
  }

  function handleDragLeave() {
    isDragOver = false;
  }

  async function handleDrop(e) {
    e.preventDefault();
    isDragOver = false;
    const files = e.dataTransfer?.files;
    if (files?.length > 0) {
      await uploadFile(files[0]);
    }
  }

  async function handleFileSelect(e) {
    const files = e.target?.files;
    if (files?.length > 0) {
      await uploadFile(files[0]);
    }
  }

  async function uploadFile(file) {
    if (!file.name.endsWith('.epub')) {
      error = 'Only EPUB files are supported';
      return;
    }

    error = null;
    isUploading = true;

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const result = await response.json();
      uploadedFile = result;
      onUpload(result.filename);
    } catch (e) {
      error = e.message;
    } finally {
      isUploading = false;
    }
  }

  function clearFile() {
    uploadedFile = null;
    onUpload(null);
  }
</script>

<div class="upload-container">
  <h3>Upload EPUB</h3>

  {#if !uploadedFile}
    <div
      class="dropzone"
      class:dragover={isDragOver}
      ondragover={handleDragOver}
      ondragleave={handleDragLeave}
      ondrop={handleDrop}
    >
      {#if isUploading}
        <span class="spinner"></span>
        <p>Uploading...</p>
      {:else}
        <p>Drop EPUB file here</p>
        <p>or</p>
        <label class="file-input-label">
          Browse files
          <input type="file" accept=".epub" onchange={handleFileSelect} />
        </label>
      {/if}
    </div>
  {:else}
    <div class="uploaded-file">
      <span class="file-icon">📚</span>
      <span class="filename">{uploadedFile.filename}</span>
      <button class="clear-btn" onclick={clearFile}>×</button>
    </div>
  {/if}

  {#if error}
    <p class="error">{error}</p>
  {/if}
</div>

<style>
  .upload-container {
    margin-bottom: 1.5rem;
  }

  h3 {
    margin-bottom: 0.5rem;
    font-size: 1rem;
    color: #333;
  }

  .dropzone {
    border: 2px dashed #ccc;
    border-radius: 8px;
    padding: 2rem;
    text-align: center;
    transition: all 0.2s;
    background: #fafafa;
  }

  .dropzone.dragover {
    border-color: #4a90d9;
    background: #f0f7ff;
  }

  .dropzone p {
    margin: 0.25rem 0;
    color: #666;
  }

  .file-input-label {
    display: inline-block;
    padding: 0.5rem 1rem;
    background: #4a90d9;
    color: white;
    border-radius: 4px;
    cursor: pointer;
    margin-top: 0.5rem;
  }

  .file-input-label:hover {
    background: #3a7fc9;
  }

  .file-input-label input {
    display: none;
  }

  .uploaded-file {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 1rem;
    background: #e8f4e8;
    border-radius: 8px;
  }

  .file-icon {
    font-size: 1.5rem;
  }

  .filename {
    flex: 1;
    font-weight: 500;
  }

  .clear-btn {
    background: none;
    border: none;
    font-size: 1.25rem;
    cursor: pointer;
    color: #666;
  }

  .clear-btn:hover {
    color: #333;
  }

  .error {
    color: #d32f2f;
    margin-top: 0.5rem;
  }

  .spinner {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 2px solid #ccc;
    border-top-color: #4a90d9;
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }
</style>
