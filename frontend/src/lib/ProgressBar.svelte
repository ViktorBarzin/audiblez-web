<script>
  let { progress = 0, status = 'pending', eta = null, currentChapter = null } = $props();

  function getStatusColor(status) {
    const colors = {
      pending: '#9e9e9e',
      processing: '#4a90d9',
      completed: '#4caf50',
      failed: '#f44336',
      cancelled: '#ff9800'
    };
    return colors[status] || '#9e9e9e';
  }

  function getStatusText(status) {
    const texts = {
      pending: 'Waiting...',
      processing: 'Converting...',
      completed: 'Complete!',
      failed: 'Failed',
      cancelled: 'Cancelled'
    };
    return texts[status] || status;
  }
</script>

<div class="progress-bar-container">
  <div class="progress-bar" style="--progress: {progress}%; --status-color: {getStatusColor(status)}">
    <div class="progress-fill"></div>
    <span class="progress-text">{Math.round(progress)}%</span>
  </div>

  <div class="progress-info">
    <span class="status" style="color: {getStatusColor(status)}">{getStatusText(status)}</span>
    {#if eta}
      <span class="eta">ETA: {eta}</span>
    {/if}
  </div>

  {#if currentChapter}
    <div class="current-chapter">{currentChapter}</div>
  {/if}
</div>

<style>
  .progress-bar-container {
    width: 100%;
  }

  .progress-bar {
    position: relative;
    height: 24px;
    background: #e0e0e0;
    border-radius: 12px;
    overflow: hidden;
  }

  .progress-fill {
    position: absolute;
    top: 0;
    left: 0;
    height: 100%;
    width: var(--progress);
    background: var(--status-color);
    transition: width 0.3s ease;
  }

  .progress-text {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-size: 0.75rem;
    font-weight: 600;
    color: #333;
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    margin-top: 0.25rem;
    font-size: 0.875rem;
  }

  .status {
    font-weight: 500;
  }

  .eta {
    color: #666;
  }

  .current-chapter {
    font-size: 0.75rem;
    color: #666;
    margin-top: 0.25rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
