#!/usr/bin/env bash
# One-time setup on a fresh Oracle Cloud Ubuntu VM (ARM). Run as the default "ubuntu" user:
#   curl -fsSL https://raw.githubusercontent.com/Nohra2005/locus-demo/main/deploy/oracle/setup.sh | bash
set -euo pipefail

echo "== Docker"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
fi

echo "== Open ports 80 and 443 (Oracle's Ubuntu image blocks them in iptables)"
for p in 80 443; do
  sudo iptables -C INPUT -p tcp --dport $p -j ACCEPT 2>/dev/null || \
    sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport $p -j ACCEPT
done
sudo apt-get install -y -qq iptables-persistent >/dev/null 2>&1 || true
sudo netfilter-persistent save >/dev/null 2>&1 || true

echo "== App files"
mkdir -p ~/locus && cd ~/locus
base=https://raw.githubusercontent.com/Nohra2005/locus-demo/main/deploy/oracle
curl -fsSLO $base/docker-compose.yml
curl -fsSLO $base/Caddyfile

IP=$(curl -fsS https://api.ipify.org)
DOMAIN="${DOMAIN:-$(echo "$IP" | tr . -).sslip.io}"
[ -f .env ] || touch .env
grep -q '^DOMAIN=' .env || echo "DOMAIN=$DOMAIN" >> .env
chmod 600 .env

echo "== Start"
sudo docker compose --env-file .env pull
sudo docker compose --env-file .env up -d
echo
echo "Done. In 2-3 minutes Locus will be at: https://$DOMAIN"
echo "Optional AI keys: add GOOGLE_API_KEY=... and GROQ_API_KEY=... to ~/locus/.env,"
echo "then run: cd ~/locus && sudo docker compose up -d"
