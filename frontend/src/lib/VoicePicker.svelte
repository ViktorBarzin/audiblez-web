<script>
  import ClonedVoicesList from './ClonedVoicesList.svelte';

  let { selectedVoice = $bindable(null) } = $props();

  let activeTab = $state('preset');
  let groupedVoices = $state({});
  let playingVoice = $state(null);
  let audioElement = $state(null);

  $effect(() => {
    fetchVoices();
  });

  async function fetchVoices() {
    try {
      const response = await fetch('/api/voices/grouped');
      if (response.ok) {
        groupedVoices = await response.json();
      }
    } catch (e) {
      console.error('Failed to fetch voices:', e);
    }
  }

  function selectVoice(voiceId) {
    selectedVoice = voiceId;
  }

  async function playVoiceSample(e, voiceId) {
    e.stopPropagation();

    if (playingVoice === voiceId) {
      // Stop playing
      if (audioElement) {
        audioElement.pause();
        audioElement = null;
      }
      playingVoice = null;
      return;
    }

    // Stop any currently playing audio
    if (audioElement) {
      audioElement.pause();
    }

    playingVoice = voiceId;
    audioElement = new Audio(`/api/voices/${voiceId}/sample`);

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

  function getGenderIcon(gender) {
    return gender === 'F' ? '♀' : '♂';
  }

  function getLanguageFlag(language) {
    const flags = {
      'American English': '🇺🇸',
      'British English': '🇬🇧',
      'Japanese': '🇯🇵',
      'Mandarin Chinese': '🇨🇳',
      'Spanish': '🇪🇸',
      'French': '🇫🇷',
      'Hindi': '🇮🇳',
      'Italian': '🇮🇹',
      'Brazilian Portuguese': '🇧🇷'
    };
    return flags[language] || '🌐';
  }
</script>

<div class="voice-picker">
  <h3>Select Voice</h3>

  <div class="tabs">
    <button
      class="tab"
      class:active={activeTab === 'preset'}
      onclick={() => { activeTab = 'preset'; }}
    >
      Preset Voices
    </button>
    <button
      class="tab"
      class:active={activeTab === 'cloned'}
      onclick={() => { activeTab = 'cloned'; }}
    >
      Cloned Voices
    </button>
  </div>

  {#if activeTab === 'preset'}
    <div class="voice-groups">
      {#each Object.entries(groupedVoices) as [language, languageVoices]}
        <div class="voice-group">
          <div class="language-header">
            <span class="flag">{getLanguageFlag(language)}</span>
            <span class="language-name">{language}</span>
          </div>
          <div class="voices-list">
            {#each languageVoices as voice}
              <div
                class="voice-item"
                class:selected={selectedVoice === voice.id}
                onclick={() => selectVoice(voice.id)}
              >
                <button
                  class="play-btn"
                  class:playing={playingVoice === voice.id}
                  onclick={(e) => playVoiceSample(e, voice.id)}
                  title="Play sample"
                >
                  {playingVoice === voice.id ? '⏹' : '▶'}
                </button>
                <span class="voice-name">{voice.name}</span>
                <span class="voice-id">{voice.id}</span>
                <span class="voice-gender" class:female={voice.gender === 'F'}>
                  {getGenderIcon(voice.gender)}
                </span>
              </div>
            {/each}
          </div>
        </div>
      {/each}
    </div>
  {:else}
    <div class="cloned-tab-content">
      <ClonedVoicesList bind:selectedVoice />
    </div>
  {/if}
</div>

<style>
  .voice-picker {
    max-height: 400px;
    overflow-y: auto;
  }

  h3 {
    margin-bottom: 0.5rem;
    font-size: 1rem;
    color: #333;
    position: sticky;
    top: 0;
    background: white;
    padding: 0.5rem 0;
    z-index: 1;
  }

  .tabs {
    display: flex;
    border-bottom: 2px solid #e0e0e0;
    margin-bottom: 0.75rem;
    position: sticky;
    top: 2.25rem;
    background: white;
    z-index: 1;
  }

  .tab {
    flex: 1;
    padding: 0.5rem 0.75rem;
    background: none;
    border: none;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 500;
    color: #666;
    transition: all 0.2s;
  }

  .tab:hover {
    color: #333;
    background: #f5f5f5;
  }

  .tab.active {
    color: #4a90d9;
    border-bottom-color: #4a90d9;
  }

  .cloned-tab-content {
    padding-top: 0.25rem;
  }

  .voice-groups {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .voice-group {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    overflow: hidden;
  }

  .language-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    background: #f5f5f5;
    font-weight: 500;
  }

  .flag {
    font-size: 1.25rem;
  }

  .voices-list {
    display: flex;
    flex-direction: column;
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

  .voice-name {
    flex: 1;
    font-weight: 500;
  }

  .voice-id {
    font-size: 0.75rem;
    color: #888;
    font-family: monospace;
  }

  .voice-gender {
    font-size: 1rem;
    color: #2196f3;
  }

  .voice-gender.female {
    color: #e91e63;
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
</style>
