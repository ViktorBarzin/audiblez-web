<script>
  import { jobs } from '../stores/jobs.js';
  import ProgressBar from './ProgressBar.svelte';

  $effect(() => {
    jobs.refresh();
    // Refresh every 5 seconds
    const interval = setInterval(() => jobs.refresh(), 5000);
    return () => clearInterval(interval);
  });

  async function downloadJob(job) {
    window.open(`/api/jobs/${job.id}/download`, '_blank');
  }

  function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleString();
  }
</script>

<div class="jobs-list">
  <h3>Conversion Jobs</h3>

  {#if $jobs.length === 0}
    <p class="empty">No jobs yet. Upload an EPUB and start converting!</p>
  {:else}
    <div class="jobs">
      {#each $jobs as job (job.id)}
        <div class="job-card" class:completed={job.status === 'completed'} class:failed={job.status === 'failed'}>
          <div class="job-header">
            <span class="job-filename">{job.filename}</span>
            <span class="job-voice">{job.voice}</span>
          </div>

          <ProgressBar
            progress={job.progress}
            status={job.status}
          />

          <div class="job-footer">
            <span class="job-date">{formatDate(job.created_at)}</span>
            {#if job.status === 'completed'}
              <button class="download-btn" onclick={() => downloadJob(job)}>
                Download
              </button>
            {/if}
            {#if job.error}
              <span class="job-error">{job.error}</span>
            {/if}
          </div>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .jobs-list {
    margin-top: 2rem;
  }

  h3 {
    margin-bottom: 1rem;
    font-size: 1rem;
    color: #333;
  }

  .empty {
    color: #666;
    text-align: center;
    padding: 2rem;
    background: #f5f5f5;
    border-radius: 8px;
  }

  .jobs {
    display: flex;
    flex-direction: column;
    gap: 1rem;
  }

  .job-card {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    background: white;
  }

  .job-card.completed {
    border-color: #4caf50;
    background: #f1f8e9;
  }

  .job-card.failed {
    border-color: #f44336;
    background: #ffebee;
  }

  .job-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
  }

  .job-filename {
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 70%;
  }

  .job-voice {
    font-size: 0.875rem;
    color: #666;
    background: #f0f0f0;
    padding: 0.125rem 0.5rem;
    border-radius: 4px;
  }

  .job-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 0.75rem;
  }

  .job-date {
    font-size: 0.75rem;
    color: #666;
  }

  .job-error {
    font-size: 0.75rem;
    color: #f44336;
  }

  .download-btn {
    padding: 0.375rem 0.75rem;
    background: #4caf50;
    color: white;
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.875rem;
  }

  .download-btn:hover {
    background: #43a047;
  }
</style>
