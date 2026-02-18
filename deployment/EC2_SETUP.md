# EC2 Production Setup Guide

This guide walks you through the manual steps to prepare a fresh Ubuntu 22.04 LTS EC2 instance for the Phishing Detection Service.

## 1. System Preparation
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io git curl certbot python3-certbot-nginx nginx
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
# Log out and log back in for docker group changes to take effect
```

## 2. Firewall Hardening (UFW)
```bash
sudo ufw allow OpenSSH
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

## 3. Clone & Environment
```bash
git clone https://github.com/your-username/Phishing2.0.git
cd Phishing2.0
cp .env.example .env
# Edit .env with your API keys
nano .env
```

## 4. Nginx Setup
```bash
sudo cp deployment/nginx/phishing.conf /etc/nginx/sites-available/phishing
sudo ln -s /etc/nginx/sites-available/phishing /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
```

## 5. SSL (Let's Encrypt)
```bash
sudo certbot --nginx -d yourdomain.com
```

## 6. Initial Manual Deploy
```bash
bash deployment/scripts/deploy.sh
```

## 7. GitHub Actions Secrets
Add these to your GitHub Repo Secrets:
- `EC2_HOST`: The IP or Domain of your EC2.
- `SSH_PRIVATE_KEY`: The contents of your `.pem` file.
- `SAFE_BROWSING_API_KEY`: Google Safe Browsing API Key.
