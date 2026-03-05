# GitHub Setup Guide for SmartVayu

This guide will help you push the SmartVayu project to GitHub.

## Prerequisites

- Git installed on your system
- GitHub account created
- Git configured with your credentials

## Step 1: Verify Git Configuration

Check if Git is configured with your name and email:

```bash
git config --global user.name
git config --global user.email
```

If not configured, set them:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

## Step 2: Create GitHub Repository

1. Go to https://github.com
2. Click the "+" icon in the top right
3. Select "New repository"
4. Fill in the details:
   - **Repository name**: `smartvayu`
   - **Description**: "Temperature Prediction & Control System with LSTM and NLP"
   - **Visibility**: Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
5. Click "Create repository"

## Step 3: Add Remote Repository

After creating the repository on GitHub, you'll see a page with setup instructions. Copy the repository URL and run:

```bash
# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/smartvayu.git
```

Or if using SSH:

```bash
git remote add origin git@github.com:YOUR_USERNAME/smartvayu.git
```

## Step 4: Verify Remote

Check that the remote was added correctly:

```bash
git remote -v
```

You should see:
```
origin  https://github.com/YOUR_USERNAME/smartvayu.git (fetch)
origin  https://github.com/YOUR_USERNAME/smartvayu.git (push)
```

## Step 5: Push to GitHub

Push your code to GitHub:

```bash
# Push to main branch (GitHub's default)
git branch -M main
git push -u origin main
```

If you encounter authentication issues, you may need to:

### Option A: Use Personal Access Token (Recommended)

1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Click "Generate new token (classic)"
3. Give it a name (e.g., "SmartVayu Project")
4. Select scopes: `repo` (full control of private repositories)
5. Click "Generate token"
6. Copy the token (you won't see it again!)
7. When pushing, use the token as your password

### Option B: Use SSH Key

1. Generate SSH key:
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   ```
2. Add SSH key to ssh-agent:
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```
3. Copy public key:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
4. Add to GitHub: Settings → SSH and GPG keys → New SSH key
5. Change remote to SSH:
   ```bash
   git remote set-url origin git@github.com:YOUR_USERNAME/smartvayu.git
   ```

## Step 6: Verify Upload

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/smartvayu`
2. You should see all your files
3. The README.md will be displayed on the main page

## Step 7: Add Repository Topics (Optional)

On your GitHub repository page:
1. Click the gear icon next to "About"
2. Add topics: `machine-learning`, `lstm`, `nlp`, `temperature-prediction`, `python`, `tensorflow`, `deep-learning`, `time-series`, `iot`, `smart-home`
3. Add description: "Temperature Prediction & Control System with LSTM and NLP"
4. Add website (if you have one)
5. Click "Save changes"

## Common Issues and Solutions

### Issue: "fatal: remote origin already exists"

**Solution:**
```bash
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/smartvayu.git
```

### Issue: Authentication failed

**Solution:** Use a Personal Access Token instead of your password (see Option A above)

### Issue: Large files warning

**Solution:** The data files might be large. If GitHub rejects them:
1. Add large files to `.gitignore`
2. Remove them from git:
   ```bash
   git rm --cached data/processed/combined_plus_sensor_data.csv
   git commit -m "Remove large data file"
   ```
3. Consider using Git LFS for large files:
   ```bash
   git lfs install
   git lfs track "*.csv"
   git add .gitattributes
   git commit -m "Add Git LFS tracking"
   ```

### Issue: Push rejected

**Solution:**
```bash
# Pull first, then push
git pull origin main --allow-unrelated-histories
git push -u origin main
```

## Next Steps

After successfully pushing to GitHub:

1. **Add a GitHub Actions workflow** for automated testing
2. **Enable GitHub Pages** to host documentation
3. **Add badges** to README (build status, license, etc.)
4. **Create releases** for version management
5. **Set up branch protection** rules
6. **Add collaborators** if working in a team

## Quick Reference Commands

```bash
# Check status
git status

# Add files
git add .

# Commit changes
git commit -m "Your commit message"

# Push to GitHub
git push origin main

# Pull from GitHub
git pull origin main

# View commit history
git log --oneline

# Create new branch
git checkout -b feature-name

# Switch branches
git checkout main
```

## Support

If you encounter issues:
- Check GitHub's documentation: https://docs.github.com
- GitHub Community: https://github.community
- Stack Overflow: https://stackoverflow.com/questions/tagged/git

---

**Congratulations!** Your SmartVayu project is now on GitHub! 🎉
