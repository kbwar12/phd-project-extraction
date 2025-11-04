# Git Repository Setup Guide

Your local git repository has been initialized and your first commit is ready!

## Next Steps: Create Remote Repository

### Option 1: GitHub (Recommended)

1. **Create a new repository on GitHub:**
   - Go to https://github.com/new
   - Choose a repository name (e.g., `phd-project-extraction`)
   - Choose Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
   - Click "Create repository"

2. **Connect your local repository to GitHub:**
   ```powershell
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```
   
   Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and the repository name you chose.

### Option 2: GitLab

1. **Create a new project on GitLab:**
   - Go to https://gitlab.com/projects/new
   - Click "Create blank project"
   - Choose a project name
   - Choose Public or Private
   - **DO NOT** initialize with README (we already have files)
   - Click "Create project"

2. **Connect your local repository to GitLab:**
   ```powershell
   git remote add origin https://gitlab.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

### Option 3: Bitbucket

1. **Create a new repository on Bitbucket:**
   - Go to https://bitbucket.org/repo/create
   - Choose a repository name
   - Choose Public or Private
   - Click "Create repository"

2. **Connect your local repository to Bitbucket:**
   ```powershell
   git remote add origin https://bitbucket.org/YOUR_USERNAME/YOUR_REPO_NAME.git
   git branch -M main
   git push -u origin main
   ```

## Authentication

If you haven't set up authentication, you may need to:

- **GitHub**: Use a Personal Access Token (Settings → Developer settings → Personal access tokens)
- **GitLab**: Use a Personal Access Token or SSH keys
- **Bitbucket**: Use an App Password (Personal settings → App passwords)

## Verify Your Setup

After pushing, verify everything worked:
```powershell
git remote -v
git status
```

Your repository should now be online and ready to use!

