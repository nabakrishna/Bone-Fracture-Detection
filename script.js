// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const imagePreview = document.getElementById('imagePreview');
const previewImage = document.getElementById('previewImage');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const analyzeBtn = document.getElementById('analyzeBtn');
const loadingContainer = document.getElementById('loadingContainer');
const resultsSection = document.getElementById('results');
const confidenceSlider = document.getElementById('confidenceSlider');
const confidenceValue = document.getElementById('confidenceValue');

// Navigation
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        
        // Remove active class from all links
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        
        // Add active class to clicked link
        link.classList.add('active');
        
        // Smooth scroll to section
        const targetId = link.getAttribute('href').substring(1);
        const targetSection = document.getElementById(targetId);
        
        if (targetSection) {
            targetSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// File Upload Handling
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileUpload(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// Handle File Upload
function handleFileUpload(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please upload an image file (JPG, PNG, JPEG)');
        return;
    }
    
    // Validate file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        fileName.textContent = file.name;
        fileSize.textContent = (file.size / (1024 * 1024)).toFixed(2) + ' MB';
        
        // Show preview and hide upload area
        uploadArea.style.display = 'none';
        imagePreview.classList.remove('hidden');
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// Change Image Button
document.getElementById('changeImage').addEventListener('click', () => {
    uploadArea.style.display = 'block';
    imagePreview.classList.add('hidden');
    analyzeBtn.disabled = true;
    fileInput.value = '';
});

// Confidence Slider
confidenceSlider.addEventListener('input', (e) => {
    confidenceValue.textContent = e.target.value;
});

// Analyze Button
analyzeBtn.addEventListener('click', () => {
    startAnalysis();
});

// Start Analysis
// function startAnalysis() {
//     // Hide detection section and show loading
//     document.getElementById('detection').style.display = 'none';
//     loadingContainer.classList.remove('hidden');
//     resultsSection.classList.add('hidden');
    
//     // Simulate analysis progress
//     let progress = 0;
//     const progressInterval = setInterval(() => {
//         progress += Math.random() * 15;
//         if (progress > 100) progress = 100;
        
//         document.getElementById('progressFill').style.width = progress + '%';
//         document.getElementById('progressText').textContent = Math.round(progress) + '%';
        
//         if (progress >= 100) {
//             clearInterval(progressInterval);
//             setTimeout(() => {
//                 showResults();
//             }, 1000);
//         }
//     }, 200);
// }

// // Show Results
// function showResults() {
//     loadingContainer.classList.add('hidden');
//     resultsSection.classList.remove('hidden');
    
//     // Simulate results (in real implementation, this would come from your Python backend)
//     const mockResults = {
//         fractureDetected: Math.random() > 0.5,
//         confidence: Math.round(60 + Math.random() * 35),
//         detections: [
//             {
//                 type: 'Potential Fracture',
//                 location: 'Radius bone',
//                 confidence: '87%',
//                 severity: 'Moderate'
//             },
//             {
//                 type: 'Analysis Area',
//                 location: 'Wrist joint',
//                 confidence: '92%',
//                 severity: 'High attention'
//             }
//         ]
//     };
    
//     displayResults(mockResults);
    
//     // Scroll to results
//     resultsSection.scrollIntoView({
//         behavior: 'smooth',
//         block: 'start'
//     });
// }

// // Display Results
// function displayResults(results) {
//     const statusIndicator = document.getElementById('statusIndicator');
//     const statusText = document.getElementById('statusText');
//     const confidenceScore = document.getElementById('confidenceScore');
//     const detectionDetails = document.getElementById('detectionDetails');
//     const originalResult = document.getElementById('originalResult');
//     const analyzedResult = document.getElementById('analyzedResult');
    
//     // Set status
//     if (results.fractureDetected) {
//         statusIndicator.className = 'status-indicator positive';
//         statusIndicator.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
//         statusText.textContent = 'Potential Fracture Detected';
//     } else {
//         statusIndicator.className = 'status-indicator negative';
//         statusIndicator.innerHTML = '<i class="fas fa-check-circle"></i>';
//         statusText.textContent = 'No Fracture Detected';
//     }
    
//     // Set confidence
//     confidenceScore.textContent = results.confidence + '%';
    
//     // Set images (using the uploaded image for both - in real app, analyzed would have bounding boxes)
//     originalResult.src = previewImage.src;
//     analyzedResult.src = previewImage.src;
    
//     // Set detection details
//     detectionDetails.innerHTML = '';
//     results.detections.forEach(detection => {
//         const detailItem = document.createElement('div');
//         detailItem.className = 'detail-item';
//         detailItem.innerHTML = `
//             <div class="detail-label">${detection.type}</div>
//             <div class="detail-value">${detection.location}</div>
//             <div class="detail-label">Confidence: ${detection.confidence}</div>
//             <div class="detail-label">Severity: ${detection.severity}</div>
//         `;
//         detectionDetails.appendChild(detailItem);
//     });
// }

// Start Analysis - UPDATED to call real backend
function startAnalysis() {
    // Hide detection section and show loading
    document.getElementById('detection').style.display = 'none';
    loadingContainer.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    
    // Get the actual file from input
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file first');
        return;
    }
    
    // Prepare form data for upload
    const formData = new FormData();
    formData.append('file', file);
    formData.append('confidence', confidenceSlider.value);
    
    // Show progress animation
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 100) progress = 100;
        
        document.getElementById('progressFill').style.width = progress + '%';
        document.getElementById('progressText').textContent = Math.round(progress) + '%';
        
        if (progress >= 100) {
            clearInterval(progressInterval);
        }
    }, 200);
    
    // Call your Flask backend
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        clearInterval(progressInterval);
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('progressText').textContent = '100%';
        
        setTimeout(() => {
            if (data.success) {
                showResults(data);  // Pass real data from backend
            } else {
                alert('Analysis failed: ' + (data.error || 'Unknown error'));
                loadingContainer.classList.add('hidden');
                document.getElementById('detection').style.display = 'block';
            }
        }, 1000);
    })
    .catch(error => {
        clearInterval(progressInterval);
        console.error('Error:', error);
        alert('Analysis failed. Please try again.');
        loadingContainer.classList.add('hidden');
        document.getElementById('detection').style.display = 'block';
    });
}

// Show Results - UPDATED to use real backend data
function showResults(backendData) {
    loadingContainer.classList.add('hidden');
    resultsSection.classList.remove('hidden');
    
    // Use real results from your Flask backend
    const results = {
        fractureDetected: backendData.fracture_detected,
        confidence: Math.round(backendData.confidence * 100),
        detections: backendData.detections || [],
        image_url: backendData.image_url,
        analyzed_image_url: backendData.analyzed_image_url,  // KEY: This has the annotated image
        analysis: backendData.analysis || {}
    };
    
    displayResults(results);
    
    // Scroll to results
    resultsSection.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
}

// Display Results - UPDATED to show different images
function displayResults(results) {
    const statusIndicator = document.getElementById('statusIndicator');
    const statusText = document.getElementById('statusText');
    const confidenceScore = document.getElementById('confidenceScore');
    const detectionDetails = document.getElementById('detectionDetails');
    const originalResult = document.getElementById('originalResult');
    const analyzedResult = document.getElementById('analyzedResult');
    
    // Set status
    if (results.fractureDetected) {
        statusIndicator.className = 'status-indicator positive';
        statusIndicator.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        statusText.textContent = 'Potential Fracture Detected';
    } else {
        statusIndicator.className = 'status-indicator negative';
        statusIndicator.innerHTML = '<i class="fas fa-check-circle"></i>';
        statusText.textContent = 'No Fracture Detected';
    }
    
    // Set confidence
    confidenceScore.textContent = results.confidence + '%';
    
    // ✅ KEY FIX: Set DIFFERENT images for left and right
    originalResult.src = results.image_url || previewImage.src;  // Original X-ray
    analyzedResult.src = results.analyzed_image_url || results.image_url || previewImage.src;  // Annotated with boxes
    
    // Force refresh if same filename (add timestamp)
    if (results.analyzed_image_url) {
        analyzedResult.src = results.analyzed_image_url + '?t=' + new Date().getTime();
    }
    
    // Set detection details using real backend data
    detectionDetails.innerHTML = '';
    if (results.detections && results.detections.length > 0) {
        results.detections.forEach(detection => {
            const detailItem = document.createElement('div');
            detailItem.className = 'detail-item';
            detailItem.innerHTML = `
                <div class="detail-label">${detection.type}</div>
                <div class="detail-value">${detection.location}</div>
                <div class="detail-label">Confidence: ${detection.confidence}</div>
                <div class="detail-label">Severity: ${detection.severity}</div>
            `;
            detectionDetails.appendChild(detailItem);
        });
    } else {
        detectionDetails.innerHTML = '<div class="detail-item">No specific detections found</div>';
    }
}

// Action Buttons
document.getElementById('downloadReport').addEventListener('click', () => {
    // In real implementation, generate and download PDF report
    alert('Report download functionality would be implemented here');
});

document.getElementById('analyzeAnother').addEventListener('click', () => {
    // Reset to detection section
    document.getElementById('detection').style.display = 'block';
    resultsSection.classList.add('hidden');
    uploadArea.style.display = 'block';
    imagePreview.classList.add('hidden');
    analyzeBtn.disabled = true;
    fileInput.value = '';
    
    // Scroll back to detection
    document.getElementById('detection').scrollIntoView({
        behavior: 'smooth',
        block: 'start'
    });
});

document.getElementById('shareResults').addEventListener('click', () => {
    // In real implementation, provide sharing options
    alert('Sharing functionality would be implemented here');
});

// Smooth scrolling for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

