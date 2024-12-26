document.addEventListener('DOMContentLoaded', () => {
  const dropZone = document.getElementById('drop-zone');
  const functionsPopup = document.getElementById('functions-popup');
  const aiPromptContainer = document.getElementById('ai-prompt-container');
  const aiPromptInput = document.getElementById('ai-prompt');
  const applyAiPromptButton = document.getElementById('apply-ai-prompt');
  const loadingOverlay = document.getElementById('loading-overlay');
  let uploadedFileId = null;
  let uploadedFileExtension = null;

  // Prevent default drag behaviors
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  // Handle drag and drop visual feedback
  dropZone.addEventListener('dragover', () => dropZone.classList.add('hover'));
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('hover'));
  dropZone.addEventListener('drop', handleDrop);
  dropZone.addEventListener('click', handleClick);

  function showLoading() {
    loadingOverlay.classList.remove('hidden');
  }

  function hideLoading() {
    loadingOverlay.classList.add('hidden');
  }


  function handleDrop(e) {
    const files = e.dataTransfer.files;
    if (files.length > 0) uploadFile(files[0]);
  }

  function handleClick() {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.onchange = e => {
      if (e.target.files.length > 0) uploadFile(e.target.files[0]);
    };
    fileInput.click();
  }

  function uploadFile(file) {
    console.log('Uploading file:', file);
    const formData = new FormData();
    formData.append('file', file);
  
    fetch('/upload', {
      method: 'POST',
      body: formData,
    })
      .then(response => {
        if (!response.ok) throw new Error('Upload failed');
        return response.json();
      })
      .then(data => {
        console.log('Upload response:', data);
        if (data.file_id) {
          uploadedFileId = data.file_id;
          uploadedFileExtension = data.extension || 'png'; // Default to PNG if undefined
          functionsPopup.classList.remove('hidden'); // Show the popup
        } else {
          console.error('Invalid upload response:', data);
          alert('Failed to process uploaded file.');
        }
      })
      .catch(error => {
        console.error('Upload error:', error);
        alert('An error occurred while uploading the file.');
      });
  }

  // Handle function tiles
  document.querySelectorAll('.tile').forEach(tile => {
    tile.addEventListener('click', () => {
      const action = tile.getAttribute('data-action');
      
      // Reset state
      aiPromptContainer.classList.add('hidden');
      
      if (action === 'extend_ai') {
        aiPromptContainer.classList.remove('hidden');
        aiPromptInput.focus();
      } else if (uploadedFileId && uploadedFileExtension) {
        processImage(action, uploadedFileId, uploadedFileExtension);
      }
    });
  });

  // Handle AI Prompt Generation
  applyAiPromptButton.addEventListener('click', () => {
    const prompt = aiPromptInput.value.trim();
    if (!prompt) {
      alert('Please enter a prompt for AI extension.');
      return;
    }
    if (uploadedFileId && uploadedFileExtension) {
      processImageWithPrompt('extend_ai', uploadedFileId, uploadedFileExtension, prompt);
    }
  });

  document.querySelectorAll('.tile-disabled').forEach(tile => {
    tile.addEventListener('click', () => {
      alert('This feature is coming soon!');
    });
  });

  function processImage(action, fileId, extension) {
    showLoading();
    const formData = new FormData();
    formData.append('file_id', fileId);
    formData.append('extension', extension);

    fetch(`/process/${action}`, {
      method: 'POST',
      body: formData,
    })
      .then(handleProcessingResponse)
      .catch(error => {
        console.error('Processing error:', error);
        alert('An error occurred while processing the image');
      })
      .finally(hideLoading);
  }

  function processImageWithPrompt(action, fileId, extension, prompt) {
    showLoading();
    const formData = new FormData();
    formData.append('file_id', fileId);
    formData.append('extension', extension);
    formData.append('prompt', prompt);

    fetch(`/process/${action}`, {
      method: 'POST',
      body: formData,
    })
      .then(handleProcessingResponse)
      .catch(error => {
        console.error('Processing error:', error);
        alert('An error occurred while processing the image');
      })
      .finally(hideLoading);
  }

  function handleProcessingResponse(response) {
    if (!response.ok) throw new Error('Processing failed');
    
    return response.blob().then(blob => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `processed_${Date.now()}.${uploadedFileExtension}`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    });
  }
});