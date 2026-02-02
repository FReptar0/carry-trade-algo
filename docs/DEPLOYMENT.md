# Deployment Guide: Carry Trade Paper Trading on EC2

## Prerequisites

1. **OANDA Practice Account**: Sign up at https://www.oanda.com/
2. **API Token**: Generate from Account Settings > Manage API Access
3. **Account ID**: Found in Account Settings (format: `101-001-XXXXXXXX-001`)
4. **AWS Account**: For EC2 deployment

## Local Testing

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env with your OANDA credentials
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Run Locally (without Docker)

```bash
uv run python scripts/run_live.py
```

### 4. Run with Docker (recommended before deploying)

```bash
docker-compose up --build
```

## EC2 Deployment

### 1. Launch Instance

- **AMI**: Ubuntu 24.04 LTS
- **Instance type**: t3.small (2 vCPU, 2 GB RAM, ~$15/month)
- **Storage**: 20 GB gp3
- **Security group**: SSH (port 22) from your IP only
- **Key pair**: Create or use existing

### 2. Connect and Install Docker

```bash
ssh -i your-key.pem ubuntu@<ec2-ip>

# Install Docker
sudo apt update && sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker ubuntu
# Log out and back in for group change
exit
ssh -i your-key.pem ubuntu@<ec2-ip>
```

### 3. Deploy the Application

```bash
# Clone your repository
git clone <your-repo-url> carry-trade-algo
cd carry-trade-algo

# Create .env with OANDA credentials
cat > .env << 'EOF'
OANDA_ACCESS_TOKEN=your-practice-api-token
OANDA_ACCOUNT_ID=101-001-XXXXXXXX-001
OANDA_ENVIRONMENT=practice
TRADING_PAIRS=USD/JPY,AUD/JPY
POSITION_SIZE_UNITS=10000
CHECK_INTERVAL_MIN=60
INITIAL_EQUITY=100000
EOF

# Build and start
docker compose up -d --build
```

### 4. Verify It's Running

```bash
# Check container status
docker ps

# View recent logs
docker logs carry-trade-algo --tail 50

# Follow logs in real time
docker logs carry-trade-algo -f
```

## Monitoring

### Log Files

```bash
# View logs on EC2
ls logs/trades/
ls logs/system/
cat logs/system/$(date +%Y-%m-%d)_system.log
```

### SQLite Queries

```bash
# Check daily results
sqlite3 data/trading.db "SELECT date, daily_pnl, daily_return FROM daily_results ORDER BY date DESC LIMIT 7;"

# Check protocol status
sqlite3 data/trading.db "SELECT status, start_date, duration_days FROM protocol_state;"

# Check equity history
sqlite3 data/trading.db "SELECT timestamp, equity, drawdown FROM equity_snapshots ORDER BY timestamp DESC LIMIT 10;"

# Check trade log
sqlite3 data/trading.db "SELECT timestamp, pair, side, entry_price, exit_price, pnl FROM trade_log ORDER BY timestamp DESC LIMIT 10;"

# Check checkpoints
sqlite3 data/trading.db "SELECT day, cumulative_return, max_drawdown, recommendation FROM checkpoints;"
```

### OANDA Web Platform

Log into https://www.oanda.com/ to verify positions and account balance match what the system reports.

## Maintenance

### Restart the Container

```bash
docker compose restart
```

The system resumes from SQLite state after restarts.

### Update the Code

```bash
cd carry-trade-algo
git pull
docker compose up -d --build
```

### Stop the System

```bash
docker compose down
```

### Update Economic Calendar

Edit `src/news/events_2026.json` with new events. The file is mounted as a read-only volume, so changes take effect on container restart.

## Troubleshooting

| Issue | Check |
|-------|-------|
| Container won't start | `docker logs carry-trade-algo` for error messages |
| No trades happening | Check if market is open; check blackout periods |
| OANDA auth error | Verify token in `.env`; tokens expire after 90 days of inactivity |
| High memory usage | Check `docker stats`; t3.small has 2 GB |
| Container restarting | Check logs for crash reason; verify SQLite isn't corrupted |

## Cost Estimate

- **EC2 t3.small**: ~$15/month (on-demand) or ~$9/month (1-year reserved)
- **EBS storage**: ~$2/month (20 GB gp3)
- **Data transfer**: Negligible (API calls are small)
- **Total**: ~$17/month
