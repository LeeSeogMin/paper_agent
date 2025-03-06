// AI Paper Writing System - Client-side JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Form validation
    const paperForm = document.getElementById('paper-form');
    if (paperForm) {
        paperForm.addEventListener('submit', function(event) {
            const researchTopic = document.getElementById('research_topic');
            if (!researchTopic.value.trim()) {
                event.preventDefault();
                showValidationError(researchTopic, 'Research topic is required');
            } else {
                // Show loading state
                const submitButton = document.querySelector('button[type="submit"]');
                if (submitButton) {
                    submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
                    submitButton.disabled = true;
                }
            }
        });
    }
    
    // File upload preview
    const fileInput = document.getElementById('file');
    const filePreview = document.getElementById('file-preview');
    if (fileInput && filePreview) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                const fileName = fileInput.files[0].name;
                filePreview.textContent = `Selected file: ${fileName}`;
                filePreview.style.display = 'block';
            } else {
                filePreview.style.display = 'none';
            }
        });
    }
    
    // Status page auto-refresh
    setupStatusRefresh();
    
    // Paper type change handler
    const paperTypeSelect = document.getElementById('paper_type');
    const additionalInstructions = document.getElementById('additional_instructions');
    if (paperTypeSelect && additionalInstructions) {
        paperTypeSelect.addEventListener('change', function() {
            const selectedType = paperTypeSelect.value;
            let placeholder = 'Enter any additional instructions or requirements for your paper';
            
            // Customize placeholder based on paper type
            switch (selectedType) {
                case 'Literature Review':
                    placeholder = 'Enter specific areas to focus on, time period to cover, or any other requirements for your literature review';
                    break;
                case 'Research Paper':
                    placeholder = 'Enter research questions, methodology preferences, or any other requirements for your research paper';
                    break;
                case 'Technical Report':
                    placeholder = 'Enter technical specifications, target audience, or any other requirements for your technical report';
                    break;
                case 'Case Study':
                    placeholder = 'Enter case details, analysis focus, or any other requirements for your case study';
                    break;
                case 'Systematic Review':
                    placeholder = 'Enter inclusion/exclusion criteria, research questions, or any other requirements for your systematic review';
                    break;
            }
            
            additionalInstructions.placeholder = placeholder;
        });
    }
});

// Show validation error for a form field
function showValidationError(element, message) {
    // Remove any existing error message
    const existingError = element.parentNode.querySelector('.invalid-feedback');
    if (existingError) {
        existingError.remove();
    }
    
    // Add error class to the element
    element.classList.add('is-invalid');
    
    // Create and append error message
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback';
    errorDiv.textContent = message;
    element.parentNode.appendChild(errorDiv);
    
    // Focus the element
    element.focus();
}

// Setup status page auto-refresh
function setupStatusRefresh() {
    const statusElement = document.getElementById('status-progress');
    if (statusElement) {
        const requestId = statusElement.dataset.requestId;
        const isComplete = statusElement.dataset.complete === 'true';
        
        if (!isComplete && requestId) {
            // Set up polling for status updates
            const pollInterval = setInterval(function() {
                fetch(`/api/status/${requestId}`)
                    .then(response => response.json())
                    .then(data => {
                        // Update progress bar
                        const progressBar = document.querySelector('.progress-bar');
                        if (progressBar) {
                            progressBar.style.width = `${data.progress_percentage}%`;
                            progressBar.setAttribute('aria-valuenow', data.progress_percentage);
                            progressBar.textContent = `${data.progress_percentage}%`;
                        }
                        
                        // Update status text
                        const statusText = document.getElementById('status-text');
                        if (statusText) {
                            statusText.textContent = data.status;
                        }
                        
                        // If complete, reload the page to show download button
                        if (data.progress_percentage >= 100) {
                            clearInterval(pollInterval);
                            location.reload();
                        }
                    })
                    .catch(error => {
                        console.error('Error fetching status:', error);
                    });
            }, 5000); // Poll every 5 seconds
        }
    }
} 