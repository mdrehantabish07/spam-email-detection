/* ==========================================================================
   Spam Email Detection System - Client JavaScript
   Handles AJAX inference, theme toggle, animations & presets
   ========================================================================== */

// Theme Management (Light / Dark Mode)
function initTheme() {
    const savedTheme = localStorage.getItem('spamguard_theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function updateThemeIcon(theme) {
    const themeBtn = document.getElementById('themeToggleBtn');
    if (!themeBtn) return;
    if (theme === 'light') {
        themeBtn.innerHTML = '<i class="fas fa-moon"></i>';
        themeBtn.setAttribute('title', 'Switch to Dark Mode');
    } else {
        themeBtn.innerHTML = '<i class="fas fa-sun"></i>';
        themeBtn.setAttribute('title', 'Switch to Light Mode');
    }
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('spamguard_theme', newTheme);
    updateThemeIcon(newTheme);
}

// Execute early before DOM render to prevent flashing
initTheme();

document.addEventListener('DOMContentLoaded', () => {
    // Attach Theme Toggle Button
    const themeBtn = document.getElementById('themeToggleBtn');
    if (themeBtn) {
        themeBtn.addEventListener('click', toggleTheme);
    }
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    updateThemeIcon(currentTheme);

    const emailInput = document.getElementById('emailInput');
    const checkBtn = document.getElementById('checkBtn');
    const clearBtn = document.getElementById('clearBtn');
    const charCount = document.getElementById('charCount');
    const wordCount = document.getElementById('wordCount');
    
    const resultPlaceholder = document.getElementById('resultPlaceholder');
    const resultContent = document.getElementById('resultContent');
    const badgeBox = document.getElementById('badgeBox');
    const badgeTitle = document.getElementById('badgeTitle');
    const badgeSubtext = document.getElementById('badgeSubtext');
    const confidenceValue = document.getElementById('confidenceValue');
    const progressBarFill = document.getElementById('progressBarFill');
    const hamProbValue = document.getElementById('hamProbValue');
    const spamProbValue = document.getElementById('spamProbValue');
    const cleanedTextVal = document.getElementById('cleanedTextVal');
    const samplePills = document.querySelectorAll('.sample-pill');

    // Preset test sample dictionary with user test cases
    const samples = {
        'lottery': "Congratulations! You have won ₹50,000 in our lucky draw. Claim your prize now by clicking this link. Offer expires today!",
        'bank': "URGENT! Your bank account will be blocked today. Verify your account immediately by clicking this link.",
        'iphone': "You have been selected to receive a FREE iPhone. Click here now to claim your prize before the offer expires.",
        'earn': "Earn ₹10,000 every day from home with zero investment. Register now and start earning immediately.",
        'lottery2': "You are the lucky winner of a ₹1,00,000 lottery prize. Send your details now to receive your money.",
        'meeting': "Hi team, our project meeting is scheduled for tomorrow at 10 AM. Please bring your progress updates.",
        'assignment': "Dear Professor, I have completed my Machine Learning assignment and will submit it before the deadline. Thank you.",
        'sync': "Hey, are we still meeting at 5 PM today? Let me know if you need to change the time.",
        'report': "Hi Rahul, I have attached the updated report. Please review it and let me know if any changes are required.",
        'personal': "Hi Mom, I will reach home around 8 PM today. Please don't wait for me for dinner."
    };

    // Update character and word count
    function updateCounts() {
        if (!emailInput) return;
        const text = emailInput.value;
        if (charCount) charCount.textContent = `${text.length} characters`;
        const words = text.trim() ? text.trim().split(/\s+/).length : 0;
        if (wordCount) wordCount.textContent = `${words} words`;
    }

    if (emailInput) {
        emailInput.addEventListener('input', updateCounts);
    }

    // Sample Pills Click Handler
    samplePills.forEach(pill => {
        pill.addEventListener('click', () => {
            const key = pill.getAttribute('data-sample');
            if (samples[key] && emailInput) {
                emailInput.value = samples[key];
                updateCounts();
                emailInput.focus();
                // Trigger prediction
                predictSpam();
            }
        });
    });

    // Clear Button Handler
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            if (emailInput) {
                emailInput.value = '';
                updateCounts();
                emailInput.focus();
            }
            if (resultPlaceholder) resultPlaceholder.style.display = 'block';
            if (resultContent) resultContent.style.display = 'none';
        });
    }

    // Predict Function
    async function predictSpam() {
        const message = emailInput.value.trim();
        if (!message) {
            alert('Please enter or paste an email message to analyze.');
            emailInput.focus();
            return;
        }

        // Set Loading State
        checkBtn.disabled = true;
        const originalBtnText = checkBtn.innerHTML;
        checkBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing Message...';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ message: message })
            });

            const data = await response.json();

            if (!data.success) {
                alert(data.error || 'Prediction failed. Please try again.');
                return;
            }

            // Display Results
            if (resultPlaceholder) resultPlaceholder.style.display = 'none';
            if (resultContent) {
                resultContent.style.display = 'block';
                resultContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }

            const isSpam = data.prediction === 'SPAM';
            const confidence = data.confidence;

            // Update Badge Box Style & Content
            if (badgeBox) {
                badgeBox.className = 'prediction-badge-box ' + (isSpam ? 'is-spam' : 'is-ham');
            }

            if (badgeTitle) {
                badgeTitle.innerHTML = isSpam 
                    ? '<i class="fas fa-exclamation-triangle"></i> 🚨 SPAM EMAIL'
                    : '<i class="fas fa-check-circle"></i> ✅ NOT SPAM (HAM)';
            }

            if (badgeSubtext) {
                badgeSubtext.textContent = isSpam
                    ? 'This message matches spam/phishing patterns identified by the trained model.'
                    : 'This message looks like legitimate, normal communication.';
            }

            // Update Confidence Progress Bar
            if (confidenceValue) confidenceValue.textContent = `${confidence}%`;
            if (progressBarFill) {
                progressBarFill.style.width = '0%';
                setTimeout(() => {
                    progressBarFill.style.width = `${confidence}%`;
                }, 50);
            }

            // Update Probabilities Breakdown
            const hamProb = data.ham_probability !== undefined ? data.ham_probability : (data.probabilities ? data.probabilities.ham : 0);
            const spamProb = data.spam_probability !== undefined ? data.spam_probability : (data.probabilities ? data.probabilities.spam : 0);
            
            if (hamProbValue) hamProbValue.textContent = `${hamProb}%`;
            if (spamProbValue) spamProbValue.textContent = `${spamProb}%`;

            // Update Preprocessed Text
            if (cleanedTextVal) {
                cleanedTextVal.textContent = data.cleaned_text || '(No text remaining)';
            }

        } catch (error) {
            console.error('Prediction request error:', error);
            alert('Could not connect to the ML backend server. Please make sure Flask is running.');
        } finally {
            checkBtn.disabled = false;
            checkBtn.innerHTML = originalBtnText;
        }
    }

    if (checkBtn) {
        checkBtn.addEventListener('click', predictSpam);
    }
});
