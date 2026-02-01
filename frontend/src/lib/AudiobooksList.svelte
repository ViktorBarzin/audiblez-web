<script>
  let audiobooks = $state([]);
  let loading = $state(true);
  let error = $state(null);
  let editingId = $state(null);
  let editingName = $state('');

  async function fetchAudiobooks() {
    try {
      const response = await fetch('/api/audiobooks');
      if (!response.ok) throw new Error('Failed to fetch audiobooks');
      audiobooks = await response.json();
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
    return (bytes / (1024 * 1024 * 1024)).toFixed(2) + ' GB';
  }

  function formatDate(timestamp) {
    return new Date(timestamp * 1000).toLocaleString();
  }

  async function downloadAudiobook(audiobook) {
    const a = document.createElement('a');
    a.href = `/api/audiobooks/${audiobook.id}/download`;
    a.download = audiobook.filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  async function deleteAudiobook(audiobook) {
    if (!confirm(`Delete "${audiobook.filename}"? This cannot be undone.`)) {
      return;
    }

    try {
      const response = await fetch(`/api/audiobooks/${audiobook.id}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error('Failed to delete audiobook');
      await fetchAudiobooks();
    } catch (e) {
      alert('Failed to delete: ' + e.message);
    }
  }

  function startRename(audiobook) {
    editingId = audiobook.id;
    // Remove extension for editing
    const name = audiobook.filename;
    const lastDot = name.lastIndexOf('.');
    editingName = lastDot > 0 ? name.substring(0, lastDot) : name;
  }

  function cancelRename() {
    editingId = null;
    editingName = '';
  }

  async function submitRename(audiobook) {
    if (!editingName.trim()) {
      alert('Name cannot be empty');
      return;
    }

    try {
      const response = await fetch(`/api/audiobooks/${audiobook.id}/rename`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ new_name: editingName })
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || 'Failed to rename');
      }

      editingId = null;
      editingName = '';
      await fetchAudiobooks();
    } catch (e) {
      alert('Failed to rename: ' + e.message);
    }
  }

  function handleKeydown(e, audiobook) {
    if (e.key === 'Enter') {
      submitRename(audiobook);
    } else if (e.key === 'Escape') {
      cancelRename();
    }
  }

  // Fetch on mount
  $effect(() => {
    fetchAudiobooks();
    const interval = setInterval(fetchAudiobooks, 30000);
    return () => clearInterval(interval);
  });

  export function refresh() {
    fetchAudiobooks();
  }
</script>

<div class="audiobooks-list">
  <div class="header">
    <h3>Completed Audiobooks</h3>
    <button class="refresh-btn" onclick={fetchAudiobooks}>Refresh</button>
  </div>

  {#if loading}
    <div class="loading">Loading audiobooks...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if audiobooks.length === 0}
    <div class="empty">No completed audiobooks yet. Convert an EPUB to get started!</div>
  {:else}
    <div class="list">
      {#each audiobooks as audiobook}
        <div class="audiobook-item">
          <div class="info">
            {#if editingId === audiobook.id}
              <div class="rename-form">
                <input
                  type="text"
                  class="rename-input"
                  bind:value={editingName}
                  onkeydown={(e) => handleKeydown(e, audiobook)}
                  autofocus
                />
                <button class="save-btn" onclick={() => submitRename(audiobook)}>Save</button>
                <button class="cancel-btn" onclick={cancelRename}>Cancel</button>
              </div>
            {:else}
              <span class="filename">{audiobook.filename}</span>
            {/if}
            <span class="meta">
              {formatSize(audiobook.size)} - {formatDate(audiobook.created_at)}
            </span>
          </div>
          <div class="actions">
            {#if editingId !== audiobook.id}
              <button class="rename-btn" onclick={() => startRename(audiobook)}>
                Rename
              </button>
            {/if}
            <button class="download-btn" onclick={() => downloadAudiobook(audiobook)}>
              Download
            </button>
            <button class="delete-btn" onclick={() => deleteAudiobook(audiobook)}>
              Delete
            </button>
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .audiobooks-list {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 1rem;
    margin-top: 2rem;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
  }

  .header h3 {
    margin: 0;
    color: #333;
  }

  .refresh-btn {
    background: #6c757d;
    color: white;
    border: none;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .refresh-btn:hover {
    background: #5a6268;
  }

  .loading, .empty {
    color: #666;
    text-align: center;
    padding: 2rem;
  }

  .error {
    color: #dc3545;
    text-align: center;
    padding: 1rem;
  }

  .list {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .audiobook-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: white;
    padding: 0.8rem 1rem;
    border-radius: 6px;
    border: 1px solid #dee2e6;
  }

  .info {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    flex: 1;
  }

  .filename {
    font-weight: 500;
    color: #333;
  }

  .meta {
    font-size: 0.85rem;
    color: #666;
  }

  .rename-form {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .rename-input {
    padding: 0.3rem 0.5rem;
    border: 1px solid #007bff;
    border-radius: 4px;
    font-size: 0.9rem;
    min-width: 200px;
  }

  .rename-input:focus {
    outline: none;
    box-shadow: 0 0 0 2px rgba(0, 123, 255, 0.25);
  }

  .save-btn {
    background: #007bff;
    color: white;
    border: none;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.85rem;
  }

  .save-btn:hover {
    background: #0056b3;
  }

  .cancel-btn {
    background: #6c757d;
    color: white;
    border: none;
    padding: 0.3rem 0.6rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.85rem;
  }

  .cancel-btn:hover {
    background: #5a6268;
  }

  .actions {
    display: flex;
    gap: 0.5rem;
  }

  .rename-btn {
    background: #007bff;
    color: white;
    border: none;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .rename-btn:hover {
    background: #0056b3;
  }

  .download-btn {
    background: #28a745;
    color: white;
    border: none;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .download-btn:hover {
    background: #218838;
  }

  .delete-btn {
    background: #dc3545;
    color: white;
    border: none;
    padding: 0.4rem 0.8rem;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .delete-btn:hover {
    background: #c82333;
  }
</style>
