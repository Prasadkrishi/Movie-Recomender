// Fetch and display dataset stats on load
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const stats = await res.json();
    document.getElementById('total_titles').textContent = stats.total_titles;
    document.getElementById('content_types').textContent = JSON.stringify(stats.content_types);
    document.getElementById('clusters').textContent = stats.clusters;
    document.getElementById('top_genres').textContent = JSON.stringify(stats.top_genres);
    document.getElementById('top_countries').textContent = JSON.stringify(stats.top_countries);
    document.getElementById('top_ratings').textContent = JSON.stringify(stats.top_ratings);
    document.getElementById('years_range').textContent =
      stats.years_range.min + ' – ' + stats.years_range.max;
  } catch (e) {
    console.error('Failed to load stats', e);
  }
}

// Fetch title suggestions as user types
let searchTimeout = null;
async function fetchSuggestions(query) {
  if (!query) {
    document.getElementById('movies-list').innerHTML = '';
    return;
  }
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    const { movies } = await res.json();
    const list = document.getElementById('movies-list');
    list.innerHTML = '';
    movies.forEach(title => {
      const opt = document.createElement('option');
      opt.value = title;
      list.appendChild(opt);
    });
  } catch (e) {
    console.error('Search error', e);
  }
}

// Handle recommendation requests
async function getRecommendations(title) {
  try {
    const res = await fetch('/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ title })
    });
    const data = await res.json();
    const ul = document.getElementById('results');
    ul.innerHTML = '';
    if (data.error) {
      ul.innerHTML = `<li class="error">${data.error}</li>`;
      return;
    }
    if (data.recommendations.length === 0) {
      ul.innerHTML = '<li>No recommendations found.</li>';
      return;
    }
    data.recommendations.forEach(rec => {
      const li = document.createElement('li');
      li.innerHTML = `
        <strong>${rec.title} (${rec.year || 'N/A'})</strong><br>
        ${rec.type} • ${rec.genre} • ${rec.duration}<br>
        Director: ${rec.director} • Country: ${rec.country}<br>
        <em>${rec.description}</em><br>
        Similarity: ${rec.similarity}
      `;
      ul.appendChild(li);
    });
  } catch (e) {
    console.error('Recommendation error', e);
  }
}

// Event hookups
window.addEventListener('DOMContentLoaded', () => {
  loadStats();

  const input = document.getElementById('movie-input');
  input.addEventListener('input', () => {
    clearTimeout(searchTimeout);
    searchTimeout = setTimeout(() => fetchSuggestions(input.value.trim()), 300);
  });

  document.getElementById('rec-btn').addEventListener('click', () => {
    const title = input.value.trim();
    if (title) getRecommendations(title);
  });
});
