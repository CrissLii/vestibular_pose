#!/bin/bash
# 生成「外网可访问」的链接（内网穿透），方便别的设备通过公网访问你的前端。
# 使用前请先确保前端、后端已在运行（如已执行 run_background.sh 或 launchd 已加载）。
#
# 用法: bash scripts/expose_tunnel.sh [端口，默认5173]

PORT="${1:-5173}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/sbin:$PATH"

echo "正在为端口 $PORT 创建公网隧道..."
echo "（确保本机已启动服务，例如 http://localhost:$PORT 可访问）"
echo ""

# 临时把 Wi‑Fi DNS 设为 1.1.1.1 并刷新缓存，退出时恢复（需本机密码）
_use_cloudflare_dns() {
  local svc old_dns old_dns_flat
  svc=$(networksetup -listallnetworkservices 2>/dev/null | grep -i 'wi-fi' | head -1 | sed 's/^[* ]*//')
  [ -z "$svc" ] && return 1
  echo "为让 cloudflared 连接 Cloudflare，将临时把 Wi‑Fi DNS 设为 1.1.1.1（需输入本机密码），结束后会自动恢复。"
  old_dns=$(networksetup -getdnsservers "$svc" 2>/dev/null)
  old_dns_flat=$(echo "$old_dns" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | tr '\n' ' ')
  sudo networksetup -setdnsservers "$svc" 1.1.1.1 8.8.8.8 2>/dev/null || return 1
  sudo dscacheutil -flushcache 2>/dev/null; sudo killall -HUP mDNSResponder 2>/dev/null
  _restore_dns() {
    if [ -n "$old_dns_flat" ]; then
      sudo networksetup -setdnsservers "$svc" $old_dns_flat 2>/dev/null
    else
      sudo networksetup -setdnsservers "$svc" "Empty" 2>/dev/null
    fi
    sudo dscacheutil -flushcache 2>/dev/null; sudo killall -HUP mDNSResponder 2>/dev/null
    echo "已恢复 Wi‑Fi 的 DNS 设置。"
  }
  trap _restore_dns EXIT
  return 0
}

_run_cloudflared() {
  # macOS：临时将 Wi‑Fi DNS 设为 1.1.1.1 并刷新缓存后再跑（会提示输入本机密码，结束后自动恢复）
  if [[ "$(uname)" == "Darwin" ]] && _use_cloudflare_dns; then
    echo "已临时将 Wi‑Fi DNS 设为 1.1.1.1 并刷新缓存，隧道结束后会自动恢复。"
    echo ""
  fi
  cloudflared tunnel --protocol http2 --url "http://127.0.0.1:$PORT"
}

if command -v cloudflared &>/dev/null; then
  echo "使用 cloudflared 创建隧道。把下面出现的 https://xxx.trycloudflare.com 发给对方即可访问。"
  echo ""
  if ! _run_cloudflared; then
    echo ""
    echo "--- 若上方出现 \"lookup _v2-origintunneld._tcp.argotunnel.com ... no such host\" ---"
    echo "多半是当前 DNS（如 198.18.x.x，常见于 VPN/代理）无法解析 Cloudflare。请任选其一："
    echo "  1) 若在使用 VPN，可暂时关闭 VPN 后再执行本脚本"
    echo "  2) 将 Mac 的 DNS 改为 1.1.1.1：系统设置 → 网络 → Wi‑Fi → 详细信息 → DNS，添加 1.1.1.1"
    echo "  3) 使用 ngrok：brew install ngrok 后执行 ngrok http $PORT"
    exit 1
  fi
elif command -v ngrok &>/dev/null; then
  echo "使用 ngrok 创建隧道。把下面出现的 https://xxx.ngrok.io 发给对方即可访问。"
  echo ""
  ngrok http "$PORT"
else
  echo "未检测到 cloudflared 或 ngrok。"
  echo ""
  echo "任选一种安装："
  echo "  brew install cloudflared   # 推荐，免费"
  echo "  brew install ngrok         # 需注册 ngrok.com 并配置 token"
  echo ""
  echo "安装后重新执行: bash $0 $PORT"
  exit 1
fi
