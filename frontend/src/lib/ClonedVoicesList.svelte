<script>
  import CreateVoiceModal from './CreateVoiceModal.svelte';

  let { selectedVoice = $bindable(null) } = $props();

  let voices = $state([]);
  let searchFilter = $state('');
  let playingVoice = $state(null);
  let audioElement = $state(null);
  let loading = $state(true);
  let error = $state(null);
  let showCreateModal = $state(false);

  let filteredVoices = $derived(
    voices.filter(v =>
      v.name.toLowerCase().includes(searchFilter.toLowerCase())
    )
  );

  $effect(() => {
    fetchVoices();
  });

  async function fetchVoices() {
    loading = true;
    error = null;

    try {
      const response = await fetch('/api/cloned-voices');
      if (!response.ok) throw new Error('Failed to fetch cloned voices');
      voices = await response.json();
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }

  function selectVoice(voiceId) {
    selectedVoice = voiceId;
  }

  async function playVoiceSample(e, voice) {
    e.stopPropagation();

    if (playingVoice === voice.id) {
      if (audioElement) {
        audioElement.pause();
        audioElement = null;
      }
      playingVoice = null;
      return;
    }

    if (audioElement) {
      audioElement.pause();
    }

    playingVoice = voice.id;
    audioElement = new Audio(`/api/cloned-voices/${voice.id}/sample`);

    audioElement.onended = () => {
      playingVoice = null;
      audioElement = null;
    };

    audioElement.onerror = () => {
      playingVoice = null;
      audioElement = null;
    };

    try {
      await audioElement.play();
    } catch (err) {
      console.error('Failed to play sample:', err);
      playingVoice = null;
      audioElement = null;
    }
  }

  async function deleteVoice(e, voice) {
    e.stopPropagation();

    if (!confirm(`Delete voice "${voice.name}"? This cannot be undone.`)) {
      return;
    }

    try {
      const response = await fetch(`/api/cloned-voices/${voice.id}`, {
        method: 'DELETE'
      });
      if (!response.ok) throw new Error('Failed to delete voice');

      // If this was the selected voice, deselect it
      if (selectedVoice === voice.id) {
        selectedVoice = null;
      }

      await fetchVoices();
    } catch (e) {
      alert('Failed to delete: ' + e.message);
    }
  }

  function getLanguageFlag(lang) {
    const flags = {
      'en': '🇺🇸',
      'zh': '🇨🇳',
      'ja': '🇯🇵',
      'ko': '🇰🇷',
      'de': '🇩🇪',
      'fr': '🇫🇷',
      'ru': '🇷🇺',
      'pt': '🇧🇷',
      'es': '🇪🇸',
      'it': '🇮🇹'
    };
    return flags[lang] || '🌐';
  }

  function getSourceBadge(source) {
    const badges = {
      'youtube': 'YT',
      'upload': 'UP',
      'recording': 'MIC'
    };
    return badges[source] || '★';
  }

  function getSourceClass(source) {
    const classes = {
      'youtube': 'source-yt',
      'upload': 'source-up',
      'recording': 'source-mic'
    };
    return classes[source] || 'source-default';
  }

  function handleVoiceCreated() {
    showCreateModal = false;
    fetchVoices();
  }
</script>

<div class="cloned-voices">
  <div class="toolbar">
    <input
      type="text"
      class="search-input"
      placeholder="Search cloned voices..."
      bind:value={searchFilter}
    />
    <button class="create-btn" onclick={() => { showCreateModal = true; }}>
      + Create New Voice
    </button>
  </div>

  {#if loading}
    <div class="loading">Loading voices...</div>
  {:else if error}
    <div class="error">{error}</div>
  {:else if filteredVoices.length === 0}
    {#if voices.length === 0}
      <div class="empty">
        <p>No cloned voices yet.</p>
        <p class="hint">Create one by searching YouTube, uploading audio, or recording your voice.</p>
      </div>
    {:else}
      <div class="empty">
        <p>No voices match "{searchFilter}"</p>
      </div>
    {/if}
  {:else}
    <div class="voices-list">
      {#each filteredVoices as voice}
        <div
          class="voice-item"
          class:selected={selectedVoice === voice.id}
          onclick={() => selectVoice(voice.id)}
        >
          <button
            class="play-btn"
            class:playing={playingVoice === voice.id}
            onclick={(e) => playVoiceSample(e, voice)}
            title="Play sample"
          >
            {playingVoice === voice.id ? '⏹' : '▶'}
          </button>
          <span class="voice-name">{voice.name}</span>
          <span class="voice-flag">{getLanguageFlag(voice.language)}</span>
          <span class="source-badge {getSourceClass(voice.source)}">{getSourceBadge(voice.source)}</span>
          <button
            class="delete-btn"
            onclick={(e) => deleteVoice(e, voice)}
            title="Delete voice"
          >
            &times;
          </button>
        </div>
      {/each}
    </div>
  {/if}
</div>

{#if showCreateModal}
  <CreateVoiceModal
    onClose={() => { showCreateModal = false; }}
    onCreated={handleVoiceCreated}
  />
{/if}

<style>
  .cloned-voices {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .toolbar {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }

  .search-input {
    flex: 1;
    padding: 0.5rem 0.75rem;
    border: 1px solid #ccc;
    border-radius: 4px;
    font-size: 0.875rem;
  }

  .search-input:focus {
    outline: none;
    border-color: #4a90d9;
    box-shadow: 0 0 0 2px rgba(74, 144, 217, 0.2);
  }

  .create-btn {
    padding: 0.5rem 0.75rem;
    background: #4caf50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 500;
    white-space: nowrap;
    transition: background 0.2s;
  }

  .create-btn:hover {
    background: #43a047;
  }

  .loading, .empty {
    text-align: center;
    padding: 1.5rem;
    color: #666;
  }

  .empty p {
    margin: 0.25rem 0;
  }

  .empty .hint {
    font-size: 0.8rem;
    color: #888;
  }

  .error {
    text-align: center;
    padding: 1rem;
    color: #f44336;
    font-size: 0.875rem;
  }

  .voices-list {
    display: flex;
    flex-direction: column;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
  }

  .voice-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    cursor: pointer;
    transition: background 0.2s;
  }

  .voice-item:hover {
    background: #f0f7ff;
  }

  .voice-item.selected {
    background: #e3f2fd;
    border-left: 3px solid #4a90d9;
  }

  .voice-item + .voice-item {
    border-top: 1px solid #f0f0f0;
  }

  .play-btn {
    width: 28px;
    height: 28px;
    border: none;
    border-radius: 50%;
    background: #4a90d9;
    color: white;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    transition: background 0.2s, transform 0.1s;
    flex-shrink: 0;
  }

  .play-btn:hover {
    background: #3a7bc8;
    transform: scale(1.1);
  }

  .play-btn.playing {
    background: #e91e63;
  }

  .play-btn.playing:hover {
    background: #c2185b;
  }

  .voice-name {
    flex: 1;
    font-weight: 500;
    font-size: 0.875rem;
  }

  .voice-flag {
    font-size: 1rem;
  }

  .source-badge {
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.125rem 0.375rem;
    border-radius: 3px;
    text-transform: uppercase;
  }

  .source-yt {
    background: #ff0000;
    color: white;
  }

  .source-up {
    background: #4a90d9;
    color: white;
  }

  .source-mic {
    background: #ff9800;
    color: white;
  }

  .source-default {
    background: #9e9e9e;
    color: white;
  }

  .delete-btn {
    width: 24px;
    height: 24px;
    border: none;
    border-radius: 50%;
    background: transparent;
    color: #999;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    transition: all 0.2s;
    flex-shrink: 0;
  }

  .delete-btn:hover {
    background: #ffebee;
    color: #f44336;
  }
</style>
