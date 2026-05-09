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
    
    // Show thumbnail
    const fileThumbnail = document.getElementById('fileThumbnail');
    const reader = new FileReader();
    reader.onload = function(e) {
        fileThumbnail.src = e.target.result;
        fileThumbnail.parentElement.parentElement.classList.remove('hidden'); // Show polaroid frame
    }
    reader.readAsDataURL(file);
    
    // Hide empty state
    document.querySelector('.pin-empty-state').style.display = 'none';
}

async function analyzeFile(file) {
    // Show analyzing state in a sticky note
    resultsContent.innerHTML = `
        <div class="sticky-note">
            <div class="pushpin blue-pin"></div>
            <h3 class="handwritten" style="text-align: center; margin-bottom: 1rem;">Analyzing Evidence...</h3>
            <div class="spinner" style="margin: 0 auto;"></div>
            <p class="handwritten-small" style="text-align: center; margin-top: 1rem;">
                Looking for forensic traces...
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
    
    // Show string connecting photo to notes
    const stringSvg = document.querySelector('.connecting-string');
    if(stringSvg) stringSvg.classList.remove('hidden');

    resultsContent.innerHTML = `
        <div class="sticky-note">
            <div class="pushpin red-pin"></div>
            <div class="result-card">
                <div class="result-header">
                    <div>
                        <div class="result-title ${isReal ? 'authentic' : 'deepfake'}">
                            ${isReal ? 'Verdict: Authentic' : 'Verdict: Deepfake'}
                        </div>
                        <p class="handwritten-small">Confidence: ${confidence}%</p>
                    </div>
                </div>
                
                <div class="indicators" style="margin-top: 1rem;">
                    <h4>Forensic Notes:</h4>
                    <ul class="indicators-list" style="list-style: none; padding: 0;">
                        ${data.indicators.map((ind, index) => `
                            <li>- ${ind}</li>
                        `).join('')}
                    </ul>
                </div>
                
                <div class="disclaimer">
                    Examine carefully. AI accuracy ~94.2%.
                </div>
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
            console.log('✅ Backend is connected and healthy');
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