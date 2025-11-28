#!/bin/bash

# Push to GitHub using SSH key
# This script assumes you have SSH key access configured

echo "MSH-QC GitHub Push Helper"
echo "=========================="

# Set default remote if not exists
echo "Setting up remote repository..."
cd /home/shared/mshqc

git remote add origin git@github.com:syahrulhidayat/mshqc.git
echo "Remote repository added: git@github.com:syahrulhidayat/mshqc.git"

# Change branch to main (GitHub default)
git branch -M main
echo "Branch renamed to main (GitHub default)"

# Set SSH key for this repository
echo "Configuring SSH key authentication..."
ssh-keygen -t rsa -b 4096 -C "syahrulhidayat" -f ~/.ssh/id_mshqc -N ""

# Display the public key to add to GitHub
echo ""
echo "======================================================================"
echo "IMPORTANT: Add the following SSH key to your GitHub account"
echo "======================================================================"
echo ""
cat ~/.ssh/id_mshqc.pub
echo ""
echo "Steps to add this SSH key to GitHub:"
echo "1. Copy the key above (including the ssh-rsa part and email)"
echo "2. Go to: https://github.com/settings/keys"
echo "3. Click 'New SSH key'"
echo "4. Paste the key and save"
echo "5. Once added, press Enter here to continue with push..."
echo "======================================================================"
echo ""

# Wait for user to add the key
read -p "Press Enter once you've added the SSH key to GitHub..."

# Configure git to use this SSH key
echo "Configuring SSH to use the new key..."
cat > ~/.ssh/config << EOF
Host github.com
  User git
  HostName github.com
  IdentityFile ~/.ssh/id_mshqc
EOF

# Set correct permissions
chmod 600 ~/.ssh/config
chmod 600 ~/.ssh/id_mshqc

# Test SSH connection
echo "Testing SSH connection to GitHub..."
ssh -T git@github.com

# Push to GitHub
echo "Pushing to GitHub..."
git push -u origin main --force

echo "Done! Your MSH-QC package is now available at:"
echo "https://github.com/syahrulhidayat/mshqc"

# Instructions for installing from GitHub
echo ""
echo "Users can now install your package with:"
echo "pip install git+https://github.com/syahrulhidayat/mshqc.git"
echo ""
echo "Or with minimal dependencies (works on all platforms):"
echo "MSHQC_WITH_LIBINT2=OFF MSHQC_WITH_LIBCINT=OFF pip install git+https://github.com/syahrulhidayat/mshqc.git"