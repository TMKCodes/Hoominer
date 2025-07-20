#!/bin/bash
# Fetches Hoominer stats and formats for HiveOS

# Fetch stats from Hoominer API
stats_raw=$(curl -s http://127.0.0.1:8042/gpu)
if [ $? -ne 0 ]; then
    echo "Failed to fetch stats from Hoominer API"
    exit 1
fi

# Check if stats_raw is empty or invalid JSON
if [ -z "$stats_raw" ] || ! echo "$stats_raw" | jq . >/dev/null 2>&1; then
    echo "Invalid or empty API response"
    exit 1
fi

# Parse JSON using jq
khs=$(echo "$stats_raw" | jq '[.hash[] / 1000] | add')     # Total hashrate in khs
hs=$(echo "$stats_raw" | jq '[.hash[] / 1000]')            # Array of per-device hashrates
busid=$(echo "$stats_raw" | jq '[.busid[] | if . == "cpu" then 0 else . end]')
air=$(echo "$stats_raw" | jq '.air')
accepted=$(echo "$stats_raw" | jq '.shares.accepted | add')
rejected=$(echo "$stats_raw" | jq '.shares.rejected | add')
hs_units="khs"
ver=$(echo "$stats_raw" | jq -r '.miner_version')
algo='hoohash'

# Calculate uptime
pid=$(pgrep -f hoominer | head -n1)
if [ -n "$pid" ]; then
    # Get uptime in seconds using ps
    uptime_seconds=$(ps -p "$pid" -o etimes= | tr -d '[:space:]')
    if [ -z "$uptime_seconds" ]; then
        uptime="0"
    else
        uptime="$uptime_seconds"
    fi
else
    uptime="0"
fi

# Format stats for HiveOS
stats=$(jq -n \
    --arg total_khs "$khs" \
    --argjson hs "$hs" \
    --arg hs_units "$hs_units" \
    --argjson temp "[]" \
    --argjson fan "[]" \
    --argjson bus_numbers "$busid" \
    --argjson accepted "$accepted" \
    --argjson rejected "$rejected" \
    --arg uptime "$uptime" \
    --arg ver "$ver" \
    --arg algo "$algo" \
    '{
        total_khs: $total_khs,
        hs: $hs,
        hs_units: $hs_units,
        temp: $temp,
        fan: $fan,
        bus_numbers: $bus_numbers,
        ar: [$accepted, $rejected],
        algo: $algo,
        uptime: $uptime,
        ver: $ver
    }')

echo $stats
