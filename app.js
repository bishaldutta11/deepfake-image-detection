// API Configuration
const API_BASE_URL = '/api';

// DOM Elements
const fileInput = document.getElementById('fileInput');
const uploadZone = document.getElementById('uploadZone');
const resultsContent = document.getElementById('resultsContent');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    initializeChart();
    checkBackendHealth();
});

function initializeEventListeners() {
    // File upload handling
    fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop functionality
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleFileDrop);
    
    // Mobile menu toggle
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const navLinks = document.getElementById('navLinks');
    
    mobileMenuBtn.addEventListener('click', function() {
        navLinks.classList.toggle('active');
    });
    
    // Navbar scroll effect
    const navbar = document.getElementById('navbar');
    window.addEventListener('scroll', function() {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });
    
    // Smooth scroll for navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
                
                // Close mobile menu if open
                navLinks.classList.remove('active');
            }
        });
    });
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        displayFileInfo(file);
        analyzeFile(file);
    }
}

function handleDragOver(e) {
    e.preventDefault();
    uploadZone.classList.add('active');
}

function handleDragLeave() {
    uploadZone.classList.remove('active');
}

function handleFileDrop(e) {
    e.preventDefault();
    uploadZone.classList.remove('active');
    
    const file = e.dataTransfer.files[0];
    if (file) {
        fileInput.files = e.dataTransfer.files;
        displayFileInfo(file);
        analyzeFile(file);
    }
}

function displayFileInfo(file) {
    fileName.textContent = file.name;
    
    // Format file size
    const sizeInMB = (file.size / (1024 * 1024)).toFixed(2);
    fileSize.textContent = `${sizeInMB} MB`;
    
    fileInfo.classList.remove('hidden');
}

async function analyzeFile(file) {
    // Show analyzing state
    resultsContent.innerHTML = `
        <div class="analyzing">
            <div class="spinner"></div>
            <div class="analyzing-text">Analyzing ${file.name}...</div>
            <p style="color: #666; text-align: center;">
                Our AI is examining facial features, lighting patterns, and other indicators of manipulation
            </p>
        </div>
    `;

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data, file.name);
        
    } catch (error) {
        console.error('Analysis error:', error);
        showError(error.message);
    }
}

function displayResults(data, filename) {
    const isReal = data.prediction === 'authentic';
    const confidence = data.confidence;
    const isRealModel = data.model_used === 'live';
    
    resultsContent.innerHTML = `
        <div class="result-card">
            <div class="result-header">
                <div class="result-icon ${isReal ? 'authentic' : 'deepfake'}">
                    <i class="fas ${isReal ? 'fa-check-circle' : 'fa-exclamation-triangle'}"></i>
                </div>
                <div>
                    <div class="result-title ${isReal ? 'authentic' : 'deepfake'}">
                        ${isReal ? '✓ Likely Authentic' : '⚠ Potential Deepfake Detected'}
                    </div>
                    <p>Analysis completed for: ${filename}</p>
                    ${isRealModel ? 
                        '<div style="color: var(--success); font-size: 0.9rem; background: #f0fff0; padding: 0.5rem; border-radius: 4px; border-left: 4px solid var(--success); margin-top: 0.5rem;">' +
                        '<i class="fas fa-check-circle"></i> Using trained AI model (MobileNetV2)</div>' : 
                        '<div style="color: var(--warning); font-size: 0.9rem; background: #fff9e6; padding: 0.5rem; border-radius: 4px; border-left: 4px solid var(--warning); margin-top: 0.5rem;">' +
                        '<i class="fas fa-info-circle"></i> Using demonstration mode</div>'
                    }
                </div>
            </div>
            
            <div class="confidence-meter">
                <div class="confidence-label">
                    <span>Confidence Level</span>
                    <span>${confidence}%</span>
                </div>
                <div class="confidence-bar">
                    <div class="confidence-fill ${isReal ? '' : 'deepfake'}" 
                         style="width: ${confidence}%;">
                        ${confidence}%
                    </div>
                </div>
            </div>
            
            <div class="indicators">
                <h4>Key Indicators:</h4>
                <ul class="indicators-list">
                    ${data.indicators.map((ind, index) => `
                        <li class="${isReal ? '' : (index < 3 ? 'deepfake' : '')}">
                            <i class="fas ${isReal ? 'fa-check' : (index < 3 ? 'fa-times' : 'fa-exclamation')}"></i>
                            ${ind}
                        </li>
                    `).join('')}
                </ul>
            </div>
            
            <div class="disclaimer">
                <strong>Note:</strong> This analysis is based on AI detection and should be verified through 
                multiple sources for critical applications. ${isRealModel ? 
                    'Model accuracy: 94.2%. Using custom-trained MobileNetV2.' : 
                    'Currently using demonstration mode with sample predictions.'
                }
            </div>
        </div>
    `;
}

function showError(message) {
    resultsContent.innerHTML = `
        <div class="result-card">
            <div style="text-align: center; color: #dc3545;">
                <i class="fas fa-exclamation-circle" style="font-size: 3rem;"></i>
                <h3>Analysis Error</h3>
                <p>${message}</p>
                <p style="font-size: 0.9rem; margin-top: 1rem;">
                    Please try again or contact support if the problem persists.
                </p>
            </div>
        </div>
    `;
}

async function checkBackendHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Backend is connected and healthy');
            console.log('🤖 Model status:', data.model_status);
        } else {
            console.warn('⚠ Backend health check failed');
        }
    } catch (error) {
        console.error('❌ Backend connection failed:', error);
    }
}

function initializeChart() {
    const ctx = document.getElementById('trendChart').getContext('2d');
    const trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['2018', '2019', '2020', '2021', '2022', '2023', '2024'],
            datasets: [{
                label: 'Detected Deepfakes (thousands)',
                data: [15, 45, 150, 450, 850, 1400, 2100],
                borderColor: '#4361ee',
                backgroundColor: 'rgba(67, 97, 238, 0.1)',
                tension: 0.4,
                fill: true,
                pointRadius: 6,
                pointHoverRadius: 8,
                pointBackgroundColor: '#4361ee',
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Growth of Deepfake Content Over Time',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Deepfakes (thousands)'
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    }
                }
            }
        }
    });
}

// Add intersection observer for animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = 1;
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

// Observe elements for animation
document.querySelectorAll('.stat-card, .info-card, .step').forEach(el => {
    el.style.opacity = 0;
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
});