document.addEventListener('DOMContentLoaded', () => {
  
  // Helper to output styled text into terminal
  function writeTerminal(text, isError = false) {
    const terminal = document.getElementById('output-terminal');
    if (!terminal) return;
    
    if (isError) {
      terminal.innerHTML = `<span style="color: #F87171;">[ERROR] ${text}</span>`;
    } else {
      terminal.innerHTML = `<pre style="color: #38BDF8; font-family: inherit; margin: 0; white-space: pre-wrap;">${text}</pre>`;
    }
  }

  // Universal helper to handle AJAX requests
  async function submitData(url, payload) {
    writeTerminal('Executing script on backend Python engine...');
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await response.json();
      
      if (data.status === 'success') {
        writeTerminal(data.output);
      } else {
        writeTerminal(data.message || 'An unknown error occurred.', true);
      }
    } catch (err) {
      writeTerminal(`Failed to connect to backend server: ${err.message}`, true);
    }
  }

  // 1. Identity Matrix Form
  const matrixForm = document.getElementById('matrix-form');
  if (matrixForm) {
    matrixForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const size = document.getElementById('matrix-size').value;
      submitData('/api/identity', { size: parseInt(size) });
    });
  }

  // 2. Probability Calculator Form
  const probForm = document.getElementById('probability-form');
  if (probForm) {
    probForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const favorable = document.getElementById('favorable-outcomes').value;
      const total = document.getElementById('total-outcomes').value;
      submitData('/api/probability', { favorable: parseFloat(favorable), total: parseFloat(total) });
    });
  }

  // 3. Random Generator Form
  const randomForm = document.getElementById('random-form');
  if (randomForm) {
    randomForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const count = document.getElementById('sample-count').value;
      const min = document.getElementById('min-val').value;
      const max = document.getElementById('max-val').value;
      submitData('/api/random', { count: parseInt(count), min: parseFloat(min), max: parseFloat(max) });
    });
  }

  // 4. Vector Operations Form
  const vectorForm = document.getElementById('vector-form');
  if (vectorForm) {
    vectorForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const vecA = document.getElementById('vector-a').value;
      const vecB = document.getElementById('vector-b').value;
      submitData('/api/vector', { vector_a: vecA, vector_b: vecB });
    });
  }

  // 5. Statistics Form
  const statsForm = document.getElementById('stats-form');
  if (statsForm) {
    statsForm.addEventListener('submit', (e) => {
      e.preventDefault();
      const dataset = document.getElementById('dataset-input').value;
      submitData('/api/statistics', { dataset: dataset });
    });
  }

  // Clear Terminal Button
  const clearBtn = document.getElementById('clear-btn');
  if (clearBtn) {
    clearBtn.addEventListener('click', () => {
      const terminal = document.getElementById('output-terminal');
      if (terminal) {
        terminal.innerHTML = '<span class="text-secondary">&gt;_ Click "Run Program" to execute script...</span>';
      }
    });
  }

  // Search Bar Filter for Index Page
  const searchInput = document.getElementById('program-search');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      const query = e.target.value.toLowerCase();
      const cards = document.querySelectorAll('.program-card-col');
      cards.forEach(card => {
        const title = card.getAttribute('data-title') ? card.getAttribute('data-title').toLowerCase() : '';
        const content = card.textContent.toLowerCase();
        if (title.includes(query) || content.includes(query)) {
          card.style.display = 'block';
        } else {
          card.style.display = 'none';
        }
      });
    });
  }

});